# pylint: disable=redefined-outer-name,unused-argument
"""Tests for the get_mazepa_pod_spec provisioning_model handling."""
from __future__ import annotations

from zetta_utils.cloud_management.resource_allocation.k8s.pod import get_mazepa_pod_spec


def _build(**kwargs):
    return get_mazepa_pod_spec(image="img", command="cmd", **kwargs)


def _required_match_expressions(pod_spec):
    """Flatten all required nodeAffinity match-expressions."""
    affinity = pod_spec.affinity
    if affinity is None or affinity.node_affinity is None:
        return []
    sel = affinity.node_affinity.required_during_scheduling_ignored_during_execution
    if sel is None:
        return []
    out = []
    for term in sel.node_selector_terms:
        for me in term.match_expressions or []:
            out.append((me.key, me.operator, list(me.values or [])))
    return out


def _preferred_match_expressions(pod_spec):
    """Flatten all preferred nodeAffinity match-expressions, with weight."""
    affinity = pod_spec.affinity
    if affinity is None or affinity.node_affinity is None:
        return []
    prefs = affinity.node_affinity.preferred_during_scheduling_ignored_during_execution or []
    out = []
    for p in prefs:
        for me in p.preference.match_expressions or []:
            out.append((p.weight, me.key, me.operator, list(me.values or [])))
    return out


def test_provisioning_string_keeps_node_selector():
    """Single-string form preserves the original nodeSelector wire format."""
    spec = _build(provisioning_model="spot")
    assert spec.node_selector == {"cloud.google.com/gke-provisioning": "spot"}
    # No provisioning constraint should leak into nodeAffinity.
    keys = [k for (k, _, _) in _required_match_expressions(spec)]
    assert "cloud.google.com/gke-provisioning" not in keys
    pref_keys = [k for (_, k, _, _) in _preferred_match_expressions(spec)]
    assert "cloud.google.com/gke-provisioning" not in pref_keys


def test_provisioning_list_uses_node_affinity():
    """List form drops the key from nodeSelector and uses nodeAffinity."""
    spec = _build(provisioning_model=["spot", "standard"])
    assert "cloud.google.com/gke-provisioning" not in spec.node_selector

    required = _required_match_expressions(spec)
    assert (
        "cloud.google.com/gke-provisioning",
        "In",
        ["spot", "standard"],
    ) in required

    preferred = _preferred_match_expressions(spec)
    assert (100, "cloud.google.com/gke-provisioning", "In", ["spot"]) in preferred


def test_provisioning_list_first_value_is_preferred():
    """First element of the list determines the preferred value."""
    spec = _build(provisioning_model=["standard", "spot"])
    preferred = _preferred_match_expressions(spec)
    assert (100, "cloud.google.com/gke-provisioning", "In", ["standard"]) in preferred


def test_provisioning_list_with_zones_merges_required_terms():
    """Provisioning + zone required terms AND together on the same NodeSelectorTerm."""
    spec = _build(
        provisioning_model=["spot", "standard"],
        required_zones=["us-central1-a", "us-central1-b"],
    )
    required = _required_match_expressions(spec)
    keys = {k for (k, _, _) in required}
    assert "cloud.google.com/gke-provisioning" in keys
    assert "topology.kubernetes.io/zone" in keys

    # Both predicates must live on the same NodeSelectorTerm so they AND.
    sel = spec.affinity.node_affinity.required_during_scheduling_ignored_during_execution
    assert len(sel.node_selector_terms) == 1
    keys_on_term = {me.key for me in sel.node_selector_terms[0].match_expressions}
    assert keys_on_term == {
        "cloud.google.com/gke-provisioning",
        "topology.kubernetes.io/zone",
    }


def test_provisioning_string_with_gpu_combines_in_node_selector():
    """gpu_accelerator_type stays in nodeSelector regardless of provisioning form."""
    spec = _build(provisioning_model="spot", gpu_accelerator_type="nvidia-l4")
    assert spec.node_selector == {
        "cloud.google.com/gke-provisioning": "spot",
        "cloud.google.com/gke-accelerator": "nvidia-l4",
    }


def test_provisioning_list_with_gpu_keeps_gpu_in_node_selector():
    spec = _build(provisioning_model=["spot", "standard"], gpu_accelerator_type="nvidia-l4")
    assert spec.node_selector == {"cloud.google.com/gke-accelerator": "nvidia-l4"}
