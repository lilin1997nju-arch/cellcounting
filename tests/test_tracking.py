from cellvision.tracking import TrackPoint, division_candidates, one_to_one_match


def test_one_to_one_does_not_reuse_child():
    parents = [TrackPoint("p1", 0, 0), TrackPoint("p2", 10, 0)]
    children = [TrackPoint("c1", 1, 0), TrackPoint("c2", 9, 0)]
    links = one_to_one_match(parents, children, maximum_distance=5)
    assert len(links) == 2
    assert len({child for _, child, _ in links}) == 2


def test_division_allows_one_to_many():
    parent = TrackPoint("p", 5, 5, area=20)
    children = [TrackPoint("c1", 4, 5, area=9), TrackPoint("c2", 6, 5, area=10)]
    assert division_candidates(parent, children, maximum_distance=3) == ["c1", "c2"]

