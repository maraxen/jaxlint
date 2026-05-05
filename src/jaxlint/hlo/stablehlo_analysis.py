"""Semantic analysis of exported StableHLO — copied from jaxbeans for standalone use."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from jax import export


@dataclass
class OperationInfo:
    """Information about a single StableHLO operation."""

    name: str
    full_name: str
    location: str | None
    result_types: list[str]
    operand_types: list[str]


@dataclass
class StableHLOGraph:
    """Semantic graph representation of exported StableHLO."""

    operations: list[OperationInfo] = field(default_factory=list)
    operation_counts: dict[str, int] = field(default_factory=dict)
    type_provenance: dict[str, Any] = field(default_factory=dict)
    source_mapping: dict[str, str] = field(default_factory=dict)


class SemanticAnalyzer:
    """Analyzes exported StableHLO for semantic properties."""

    def __init__(self, exported: export.Exported) -> None:
        self.exported = exported
        self._mlir_text: str | None = None

    @property
    def mlir_text(self) -> str:
        if self._mlir_text is None:
            self._mlir_text = self.exported.mlir_module()
        return self._mlir_text

    def get_operation_inventory(self) -> dict[str, int]:
        operations = re.findall(r"stablehlo\.(\w+)", self.mlir_text)

        counts: dict[str, int] = {}
        for op in operations:
            full_name = f"stablehlo.{op}"
            counts[full_name] = counts.get(full_name, 0) + 1

        return counts

    def get_operations(self) -> list[OperationInfo]:
        operations: list[OperationInfo] = []

        stablehlo_ops = re.finditer(
            r"%\d+(?::\d+)?\s+=\s+(stablehlo\.\w+)\s*([^:]+?)\s*:",
            self.mlir_text,
        )

        for match in stablehlo_ops:
            full_name = match.group(1)
            types_str = match.group(2).strip()

            location = self._extract_location(match.group(0))

            operations.append(OperationInfo(
                name=full_name.split(".")[-1],
                full_name=full_name,
                location=location,
                result_types=self._parse_types(types_str),
                operand_types=[],
            ))

        return operations

    def _extract_location(self, op_text: str) -> str | None:
        loc_match = re.search(r'loc\("([^"]+)"\)', op_text)
        if loc_match:
            return loc_match.group(1)

        loc_call_match = re.search(r'loc\(([^)]+)\)', op_text)
        if loc_call_match:
            return loc_call_match.group(1)

        return None

    def _parse_types(self, type_str: str) -> list[str]:
        types = []
        for tensor_match in re.finditer(r"tensor<([^>]+)>", type_str):
            types.append(f"tensor<{tensor_match.group(1)}>")
        return types

    def get_type_provenance(self) -> dict[str, Any]:
        provenance: dict[str, Any] = {}

        inputs = []
        for i, aval in enumerate(self.exported.in_avals):
            inputs.append({
                "name": f"arg{i}",
                "shape": str(aval.shape),
                "dtype": str(aval.dtype),
                "abstract": repr(aval),
            })

        outputs = []
        for i, aval in enumerate(self.exported.out_avals):
            outputs.append({
                "name": f"result{i}",
                "shape": str(aval.shape),
                "dtype": str(aval.dtype),
                "abstract": repr(aval),
            })

        provenance["inputs"] = inputs
        provenance["outputs"] = outputs

        return provenance

    def get_source_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}

        op_refs = {}
        for match in re.finditer(r'(stablehlo\.\w+).*loc\(#loc(\d+)\)', self.mlir_text):
            op_refs[match.group(1)] = f"#loc{match.group(2)}"

        for ref_match in re.finditer(r'#loc(\d+)\s+=\s+loc\("([^"]+)"', self.mlir_text):
            loc_id = f"#loc{ref_match.group(1)}"
            loc_value = ref_match.group(2)
            for op_name, ref_loc in op_refs.items():
                if ref_loc == loc_id:
                    mapping[op_name] = loc_value

        return mapping

    def parse(self) -> StableHLOGraph:
        return StableHLOGraph(
            operations=self.get_operations(),
            operation_counts=self.get_operation_inventory(),
            type_provenance=self.get_type_provenance(),
            source_mapping=self.get_source_mapping(),
        )

    def get_summary(self) -> dict[str, Any]:
        graph = self.parse()
        return {
            "operation_count": sum(graph.operation_counts.values()),
            "unique_operations": len(graph.operation_counts),
            "operations": graph.operation_counts,
            "input_types": [inp["dtype"] for inp in graph.type_provenance.get("inputs", [])],
            "output_types": [out["dtype"] for out in graph.type_provenance.get("outputs", [])],
            "has_source_mapping": len(graph.source_mapping) > 0,
        }
