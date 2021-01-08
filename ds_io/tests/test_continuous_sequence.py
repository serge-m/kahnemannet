import random

from kahnemannet.ds_io.continuous_sequence import ContinuousSequenceSampler, RandIntSampler

DUMMY_DATA_SOURCE = []


def test_1():
    groups = [0, 1, 5], [2, 3], [4]
    for seed in range(1000):
        rs = random.Random(seed)
        sampler = ContinuousSequenceSampler(DUMMY_DATA_SOURCE, groups, RandIntSampler(rs, 2, 3), random_state=rs)
        assert sorted(iter(sampler)) == sorted(range(6))


def test_single_group():
    groups = [list(range(5))]
    rs = random.Random()
    sampler = ContinuousSequenceSampler(DUMMY_DATA_SOURCE, groups, RandIntSampler(rs, 1, 5))
    assert sorted(iter(sampler)) == sorted(range(5))


def test_deterministic_sampler():
    groups = [[10, 20, 30], [1, 2, 3]]

    def sampler():
        return 3

    sampler = ContinuousSequenceSampler(DUMMY_DATA_SOURCE, groups, sampler)
    result = list(sampler)
    part1_sorted = sorted(result[:3])
    part2_sorted = sorted(result[3:])
    assert [part1_sorted, part2_sorted] == [[10, 20, 30], [1, 2, 3]] or \
           [part1_sorted, part2_sorted] == [[1, 2, 3], [10, 20, 30]]
