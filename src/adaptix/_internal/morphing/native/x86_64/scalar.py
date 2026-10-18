from dataclasses import dataclass

from adaptix._internal.morphing.native import descriptions
from adaptix._internal.morphing.native.registry import get_program as get_description
from adaptix._internal.morphing.native.runtime import decimal_error, retry_scalar


def get_program(function):
    description = get_description(function)
    if description is None or hasattr(description, "emit"):
        return description
    if isinstance(description, descriptions.ScalarProgram):
        return ScalarProgram(**vars(description))
    if isinstance(description, descriptions.DecimalProgram):
        return DecimalProgram(**vars(description))
    return None



@dataclass(frozen=True)
class DecimalProgram(descriptions.DecimalProgram):
    def emit(self, c):
        import ctypes  # noqa: PLC0415
        from decimal import Decimal  # noqa: PLC0415

        if self.loading and not hasattr(ctypes.pythonapi, "PyObject_CallOneArg"):
            return c.call(self.function, c.data)
        generic, done = c.label("decimal_slow"), c.label("decimal_done")
        result = c.slot(owned=True)
        if self.loading:
            text = c.label("decimal_text")
            c.branch_if_type(c.data, str, text)
            c.branch_if_type(c.data, Decimal, generic, invert=True)
            value = c.copy_reference(c.data)
            c.transfer(value, result)
            c.asm.jump(done)
            c.asm.label(text)
            value = c.guarded_c_call(
                "PyObject_CallOneArg", (c.constant(Decimal), c.data), decimal_error, c.data,
            )
        else:
            c.branch_if_type(c.data, Decimal, generic, invert=True)
            value = c.call_c("PyObject_Str", c.data)
        c.transfer(value, result)
        c.asm.jump(done)
        return c.fallback_result(result, self.function, generic, done)


@dataclass(frozen=True)
class ScalarProgram(descriptions.ScalarProgram):
    supports_consuming = True

    def emit(self, c, *, consume=False):
        accepted, generic, done = (c.label(prefix) for prefix in ("scalar_ok", "scalar_slow", "scalar_done"))
        result = c.data if consume else c.slot(owned=True)
        function = descriptions.resolve_function(self.function)
        integer = c.label("scalar_integer") if self.coerce_int else None
        c.asm.load(0, c.data)
        if self.nullable:
            c.load(11, c.constant(None))
            c.asm.emit(b"\x4c\x39\xd8", "check nullable scalar")
            c.asm.jump(accepted, "z")
        c.asm.emit(b"\x48\x8b\x50" + bytes([c.layout.object_type]), "load scalar type once")
        if self.coerce_int:
            c.load(11, c.constant(int))
            c.asm.emit(b"\x4c\x39\xda", "check integer coercion")
            c.asm.jump(integer, "z")
        for cls in self.types:
            c.load(11, c.constant(cls))
            c.asm.emit(b"\x4c\x39\xda", "check accepted scalar type")
            c.asm.jump(accepted, "z")
        c.asm.jump(generic)
        if self.coerce_int:
            c.asm.label(integer)
            value = convert_integer_float(c, function)
            if consume:
                c.release(c.data, nonnull=True)
            c.transfer(value, result)
            c.asm.jump(done)
        c.asm.label(accepted)
        if not consume:
            value = c.copy_reference(c.data)
            c.transfer(value, result)
        c.asm.jump(done)
        c.asm.label(generic)
        value = c.call(function, c.data)
        if consume:
            c.release(c.data, nonnull=True)
        c.transfer(value, result)
        c.asm.label(done)
        return result




def convert_integer_float(c, function):
    return c.guarded_c_call("PyNumber_Float", (c.data,), retry_scalar, c.constant(function), c.data)
