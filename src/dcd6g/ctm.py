from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceIndicators:
    device_state_mutation: bool = False
    single_node_feature_anomaly: bool = False
    collaboration_link_normal: bool = True
    abnormal_spatiotemporal_evolution: bool = False
    edge_weight_jump: bool = False
    attribution_path_deviation: bool = False
    semantic_contradiction: bool = False
    semantic_dependency_anomaly: bool = False
    explanation_dependency_anomaly: bool = False


@dataclass(frozen=True)
class TraceResult:
    alarm: bool
    label: str
    fracture: str
    response: str


def trace(trigger_source: str, indicators: TraceIndicators) -> TraceResult:
    """Implement the CTM rule ordering from the supplementary algorithm."""
    if trigger_source == "root_cause_layer" or indicators.device_state_mutation:
        return TraceResult(
            True,
            "Environmental and Verifiable Causes Tampering",
            "S->X",
            "Verify sensor data, compare the dynamic baseline, and trace interference sources.",
        )

    if indicators.semantic_dependency_anomaly:
        if indicators.single_node_feature_anomaly and indicators.collaboration_link_normal:
            return TraceResult(
                True,
                "Perception Input Poisoning",
                "X->Y",
                "Validate firmware integrity and locate malicious data injection nodes.",
            )
        if indicators.abnormal_spatiotemporal_evolution or indicators.edge_weight_jump:
            return TraceResult(
                True,
                "Decision Logic Hijacking",
                "X->Y",
                "Backtrack distributed model updates and isolate tampered dependencies.",
            )

    if indicators.explanation_dependency_anomaly:
        if indicators.attribution_path_deviation:
            return TraceResult(
                True,
                "Decision Logic Obfuscation",
                "Y->E",
                "Run explainer self-tests and restore the regular attribution path.",
            )
        if indicators.semantic_contradiction:
            return TraceResult(
                True,
                "Explanation Interface Misdirection",
                "Y->E",
                "Verify explanation consistency and isolate forged explanation outputs.",
            )

    return TraceResult(
        False,
        "Adaptive Environmental Fluctuation",
        "none",
        "Refresh the rolling causal baseline and keep monitoring.",
    )

