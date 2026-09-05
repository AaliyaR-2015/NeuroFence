"""
Unit tests for the Week 1 adversarial fuzzer scaffolding. No model or
network access required.
"""

from src.adversarial_fuzzer import (
    _TRIGGER_CANDIDATES,
    generate_batch,
)


def test_batch_has_requested_category_counts():
    batch = generate_batch(n_benign=20, n_edge_case=10, n_trigger_candidate=10, seed=1)
    assert len(batch) == 40
    counts = {"benign": 0, "edge_case": 0, "trigger_candidate": 0}
    for p in batch:
        counts[p.category] += 1
    assert counts == {"benign": 20, "edge_case": 10, "trigger_candidate": 10}


def test_batch_is_deterministic_given_a_seed():
    batch_a = generate_batch(n_benign=5, n_edge_case=5, n_trigger_candidate=5, seed=123)
    batch_b = generate_batch(n_benign=5, n_edge_case=5, n_trigger_candidate=5, seed=123)
    assert [p.as_dict() for p in batch_a] == [p.as_dict() for p in batch_b]


def test_different_seeds_produce_different_batches():
    batch_a = generate_batch(n_benign=10, n_edge_case=10, n_trigger_candidate=10, seed=1)
    batch_b = generate_batch(n_benign=10, n_edge_case=10, n_trigger_candidate=10, seed=2)
    texts_a = [p.text for p in batch_a]
    texts_b = [p.text for p in batch_b]
    assert texts_a != texts_b


def test_trigger_candidates_actually_contain_a_known_trigger():
    batch = generate_batch(n_benign=0, n_edge_case=0, n_trigger_candidate=15, seed=7)
    for p in batch:
        assert any(trigger in p.text for trigger in _TRIGGER_CANDIDATES), (
            f"trigger_candidate prompt did not contain any known trigger: {p.text!r}"
        )


def test_edge_cases_do_not_crash_on_empty_or_huge_input():
    batch = generate_batch(n_benign=0, n_edge_case=50, n_trigger_candidate=0, seed=99)
    # Just proving generation never raises and produces plain strings.
    for p in batch:
        assert isinstance(p.text, str)


if __name__ == "__main__":
    test_batch_has_requested_category_counts()
    test_batch_is_deterministic_given_a_seed()
    test_different_seeds_produce_different_batches()
    test_trigger_candidates_actually_contain_a_known_trigger()
    test_edge_cases_do_not_crash_on_empty_or_huge_input()
    print("All adversarial_fuzzer tests passed.")
