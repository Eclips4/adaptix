from dataclasses import dataclass, field
from typing import Literal

import pytest

from adaptix import P, Retort, bound, loader
from adaptix._internal.code_tools.x86_64_compiler import X86Compiler
from adaptix._internal.morphing.native.codegen import NativeCodeGenRequest
from adaptix._internal.morphing.native.plan import (
    DictPlan,
    IterablePlan,
    ModelDumperPlan,
    ModelLoaderPlan,
    ScalarPlan,
    TuplePlan,
    UnionPlan,
)
from adaptix._internal.provider.value_provider import ValueProvider


@dataclass
class Example:
    count: int
    values: list[int]


@dataclass
class FirstVariant:
    kind: Literal["first"]
    value: int


@dataclass
class SecondVariant:
    kind: Literal["second"]
    text: str


@dataclass
class Pair:
    left: Example
    right: Example


@dataclass
class RepeatedContainers:
    left: list[int]
    right: list[int]


@dataclass
class RepeatedOptional:
    left: Example | None
    right: Example | None


@dataclass
class UnhashableLoader:
    offset: int

    def __call__(self, data):
        return data + self.offset


@dataclass
class PortableGenerator:
    """Test backend for plain model fields; containers keep their fallbacks.

    Deliberately unhashable to verify that DI dependencies are cached by identity.
    No instruction encoder or native executable memory is used by this backend.
    """

    offset: int = 0
    calls: list[str] = field(default_factory=list)

    def compile(self, plan, *, hook):
        if isinstance(plan, ModelLoaderPlan):
            self.calls.append("model_loader")

            def load(data):
                result = plan.shape.constructor(
                    **{
                        param.name: plan.field_loaders[param.field_id](data[param.field_id])
                        for param in plan.shape.params
                    },
                )
                if isinstance(result, Example):
                    result.count += self.offset
                return result

            return load
        if isinstance(plan, ModelDumperPlan):
            self.calls.append("model_dumper")
            return lambda data: {
                fld.id: plan.fields_dumpers[fld.id](getattr(data, fld.id)) for fld in plan.shape.fields
            }
        if isinstance(plan, IterablePlan):
            self.calls.append("set" if plan.origin in (set, frozenset) else "list")
        else:
            self.calls.append(
                {ScalarPlan: "scalar", TuplePlan: "tuple", DictPlan: "dict", UnionPlan: "union"}[type(plan)],
            )
        return plan.fallback


def forbid_x86_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Provider bypassed the injected code generator")

    monkeypatch.setattr(X86Compiler, "__init__", forbidden)




def test_located_dependencies_and_cached_models_do_not_share_generators(monkeypatch):
    forbid_x86_compilation(monkeypatch)
    left, right = PortableGenerator(offset=10), PortableGenerator(offset=100)
    retort = Retort(
        recipe=[
            bound(P[Pair].left, ValueProvider(NativeCodeGenRequest, left)),
            bound(P[Pair].right, ValueProvider(NativeCodeGenRequest, right)),
        ],
    )
    item = {"count": 1, "values": [2]}
    assert retort.load({"left": item, "right": item}, Pair) == Pair(Example(11, [2]), Example(101, [2]))
    assert "model_loader" in left.calls
    assert "model_loader" in right.calls






def test_cached_scalar_generation_accepts_unhashable_callables(monkeypatch):
    forbid_x86_compilation(monkeypatch)
    generator = PortableGenerator()
    retort = Retort(recipe=[ValueProvider(NativeCodeGenRequest, generator), loader(int, UnhashableLoader(10))])
    assert retort.load(1, int) == 11










