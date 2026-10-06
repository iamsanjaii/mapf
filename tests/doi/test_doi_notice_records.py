from src.doi.belief import BeliefState
from src.doi.crdt import NoticeRecord, NoticeSet

A = NoticeRecord("n0", 5, "wave 2 is all in the south bays")
B = NoticeRecord("n1", 5, "the north bays are finished after this wave")
A_LATER = NoticeRecord("n0", 9, "a different wording")


def _set(*recs):
    s = NoticeSet()
    for r in recs:
        s.add(r)
    return s


def test_merge_is_commutative_associative_idempotent():
    ab = _set(A)
    ab.merge(_set(B))
    ba = _set(B)
    ba.merge(_set(A))
    assert ab.canonical() == ba.canonical() and ab.records() == [A, B]
    again = ab.copy()
    assert again.merge(ab) is False and again.canonical() == ab.canonical()
    left = _set(A)
    left.merge(_set(B))
    left.merge(_set(A_LATER))
    right = _set(B)
    right.merge(_set(A_LATER))
    right.merge(_set(A))
    assert left.canonical() == right.canonical()


def test_same_id_with_different_content_converges_on_one_version():
    one, two = _set(A), _set(A_LATER)
    one.merge(_set(A_LATER))
    two.merge(_set(A))
    assert one.get("n0") == A and two.get("n0") == A          # the smaller (tick, text) wins on both sides
    assert one.get("missing") is None


def test_belief_carries_notices_in_full_and_delta_gossip():
    sender, full, delta = BeliefState(0, {}), BeliefState(1, {}), BeliefState(2, {})
    assert sender.units() == 0 and sender.version == 0        # an empty notice set adds nothing
    before = sender.version
    sender.add_notice(A)
    assert sender.version == before + 1 and sender.units() == 1
    full.merge(sender.snapshot())
    delta.merge(sender.delta_since(before))
    assert full.notices.records() == [A] and delta.notices.records() == [A]
    assert sender.delta_since(sender.version).units() == 0
