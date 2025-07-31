import perceval as pcvl

from functools import lru_cache
from math import comb
from sympy.utilities.iterables import multiset_permutations


@lru_cache
def thresholded_size(m: int, n: int, min_n: int) -> int:
    """Size of the returned tensor. This is the number of possible output states"""
    # For thresholded output, this is the number of binary numbers having at least self.postselect 1s
    s = 0
    for k in range(min_n, n + 1):
        s += comb(m, k)
    return s


@lru_cache
def pnr_size(m: int, n: int, min_n: int) -> int:
    return comb(m + n - 1, n)


@lru_cache  # Always the same, no need to compute it each time
def generate_states(m: int, n: int, min_n: int) -> list:
    """Generate a list of all possible output states"""
    res = []
    for k in range(min_n, n + 1):
        res += _generate_state_list_k(m, k)

    return res


@lru_cache
def generate_states_multi_photons(modes_count, photons_count):
    states = []
    partitions = []

    # Generate all partitions of n into m parts
    def generate_partitions(n, m, current):
        if m == 0:
            return
        if sum(current) == photons_count or n == 0:
            current += [0] * (modes_count - len(current))
            partitions.append(current)
            return
        start = current[-1] if current else n
        for i in range(0, start + 1):
            generate_partitions(n - i, m - 1, current + [i])

    generate_partitions(photons_count, modes_count, [])
    # Generate all partitions and convert them to permutations
    for partition in partitions:
        for state in multiset_permutations(partition):
            states += [pcvl.BasicState("|" + ",".join(map(str, state)) + ">")]

    return states


@lru_cache
def _generate_state_list_k(m: int, k: int) -> list:
    """Generate all binary states of size self.m having exactly *k* 1s"""
    return list(
        map(
            pcvl.BasicState,
            pcvl.utils.qmath.distinct_permutations(k * [1] + (m - k) * [0]),
        )
    )
