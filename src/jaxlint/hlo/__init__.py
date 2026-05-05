"""Optional HLO / export analysis (`pip install jaxlint[hlo]`)."""

from jaxlint.hlo.graph_audit import GraphAuditSensor
from jaxlint.hlo.stablehlo_analysis import SemanticAnalyzer, StableHLOGraph

__all__ = ["GraphAuditSensor", "SemanticAnalyzer", "StableHLOGraph"]
