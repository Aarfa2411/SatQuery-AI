"""
Agentic Orchestration & Planning Metrics
Measures tool routing accuracy, plan success rates, and unnecessary tool calls.
"""

from typing import List, Dict, Set


def compute_tool_selection_metrics(
    invoked_tools: List[str],
    expected_tools: List[str]
) -> Dict[str, float]:
    """
    Evaluates agent tool-selection precision, recall, and unnecessary tool invocation rate.
    """
    invoked_set: Set[str] = set(invoked_tools)
    expected_set: Set[str] = set(expected_tools)

    if not expected_set:
        return {
            "tool_precision": 1.0 if not invoked_set else 0.0,
            "tool_recall": 1.0,
            "tool_f1": 1.0 if not invoked_set else 0.0,
            "unnecessary_tool_rate": 0.0 if not invoked_set else 1.0,
        }

    tp = len(invoked_set & expected_set)
    fp = len(invoked_set - expected_set)
    fn = len(expected_set - invoked_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    unnecessary_rate = fp / len(invoked_set) if len(invoked_set) > 0 else 0.0

    return {
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "tool_f1": round(f1, 4),
        "unnecessary_tool_rate": round(unnecessary_rate, 4),
    }


def compute_plan_success_rate(statuses: List[str]) -> float:
    """
    Computes fraction of executions that completed with status 'ok' or 'real_model'.
    """
    if not statuses:
        return 0.0
    successful = sum(1 for s in statuses if s in ["ok", "real_model", "domain_adapted_model", "classical_algorithm"])
    return round(float(successful / len(statuses)), 4)
