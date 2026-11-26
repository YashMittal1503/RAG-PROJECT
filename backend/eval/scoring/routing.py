"""Routing metrics: did the system take the expected path and pick the expected document?"""

from __future__ import annotations


def score_routing(
    expected_route: str,
    route: str | None,
    expected_doc: str | None,
    chosen_doc: str | None,
) -> dict:
    route_correct = None if route is None else float(route == expected_route)
    doc_correct = None
    if expected_doc is not None:
        doc_correct = float(chosen_doc == expected_doc)
    return {"route_correct": route_correct, "doc_correct": doc_correct}


def score_classification(
    expected_scope: str | None,
    scope: str | None,
    expected_aggregation: bool | None,
    aggregation: bool | None,
) -> dict:
    """Query classification accuracy, only for items that declare what to expect.
    A prediction with no classification (e.g. an error before classify_query) counts as wrong."""
    return {
        "scope_correct": None if expected_scope is None else float(scope == expected_scope),
        "aggregation_correct": None if expected_aggregation is None else float(aggregation is expected_aggregation),
    }
