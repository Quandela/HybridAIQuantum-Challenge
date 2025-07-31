"""
This module provides functionality to calculate output probability distributions for photonic quantum circuits.
It implements a fully differentiable computation of the complete output probability distribution
over all possible output Fock states, enabling gradient-based optimization of quantum circuits.

Authors:
   Cassandre Notton
   Jean Senellart

"""

import math
import torch

from typing import List, Tuple, Callable, Optional


def threshold_mapping(state: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Maps photon numbers to binary detection events (0 or 1).

    Args:
        state (Tuple[int, ...]): Input Fock state configuration tuple

    Returns:
        Tuple[int, ...]: Mapped state with photon numbers thresholded to 0 or 1
    """
    return tuple(min(count, 1) for count in state)


def pytorch_slos_output_distribution_legacy(
    unitary: torch.Tensor,
    input_state: list,
    output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
    no_bunching: bool = False,
    keep_keys: bool = True,
) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
    """
    Computes output probability distribution for a photonic quantum circuits, supporting batched unitaries.

    Torch compatible implementation based on the SLOS_full algorithm from:
    "Strong Simulation of Linear Optical Processes"
    https://arxiv.org/pdf/2206.10549

    Args:
        unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m]
        input_state (list): List of integers specifying number of photons in each input mode
        output_map_func (callable, optional): Function that maps output states to new states
            or None. If the function returns None for a state, that state is discarded.
            Default is None (no mapping).
        no_bunching (bool): If True, the algorithm is optimized for no-bunching states only (a bit redundant with
            the output_map_func, but more efficient). Default is False.
        keep_keys (bool): If True, the output state keys are returned (small overhead). Default is True.

    Returns:
        Tuple[List[Tuple[int, ...]], torch.Tensor]:
            - List of tuples representing the mapped Fock state configurations
            - For single unitary: Probability distribution tensor [num_states]
            - For batched unitaries: Probability distributions tensor [batch_size x num_states]

        Probabilities are renormalized if any states are discarded.

    Raises:
        ValueError: If the unitary matrix is not square
        ValueError: If the input state length doesn't match the unitary matrix dimension
        ValueError: If any photon number in the input state is negative

    Note:
        The function maintains differentiability with respect to the unitary parameters

    """
    if len(unitary.shape) == 2:
        is_batched = False
        unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
    else:
        is_batched = True

    batch_size, m, m2 = unitary.shape
    if m != m2:
        raise ValueError("Unitary matrices must be square")

    if len(input_state) != m:
        raise ValueError("Input state length must match unitary matrix dimension")

    if any(n < 0 for n in input_state) or sum(input_state) == 0:
        raise ValueError("Photon numbers cannot be negative or all zeros")

    if no_bunching and not all(x in [0, 1] for x in input_state):
        raise ValueError(
            "Input state must be binary (0s and 1s only) in non-bunching mode"
        )

    # Create input mode indices
    idx_n = []
    n = 0
    for i, count in enumerate(input_state):
        for _ in range(count):
            idx_n.append(i)
            n += 1

    U_F = torch.ones((batch_size, 1), dtype=torch.cfloat)
    last_combinations = {tuple([0] * m): 0}

    input_state_tensor = torch.zeros(m, dtype=torch.float)

    keys = None

    for idx, p in enumerate(idx_n):
        input_state_tensor[p] += 1
        U_Fp1 = torch.zeros(
            (
                batch_size,
                no_bunching and math.comb(m, idx + 1) or math.comb(m + idx, idx + 1),
            ),
            dtype=torch.cfloat,
        )
        combinations = {}

        for idx_amplitude, state in enumerate(last_combinations):
            amplitude = U_F[:, idx_amplitude]
            nstate = list(state)
            for i in range(m):
                if nstate[i] and no_bunching:
                    continue
                nstate[i] += 1
                nstate_tuple = tuple(nstate)
                index = combinations.get(nstate_tuple, None)
                if index is None:
                    index = combinations[nstate_tuple] = len(combinations)
                U_Fp1[:, index] += (
                    unitary[:, i, p]
                    * amplitude
                    * torch.sqrt(torch.tensor(nstate[i]) / input_state_tensor[p])
                )
                nstate[i] -= 1

        U_F = U_Fp1
        last_combinations = combinations
        if idx == len(idx_n) - 1 and keep_keys or output_map_func is not None:
            keys = combinations

    # Calculate initial probabilities
    probabilities = (U_Fp1.abs() ** 2).real

    # Apply output mapping if provided
    if output_map_func is not None:
        # Create dictionary to accumulate probabilities for mapped states
        mapped_probs_dict = {}
        mapped_keys = []
        discarded_states = False

        for idx, key in enumerate(keys):
            prob = probabilities[:, idx]
            mapped_state = output_map_func(key)
            if mapped_state is not None:  # Only include non-discarded states
                if mapped_state not in mapped_probs_dict:
                    mapped_probs_dict[mapped_state] = prob
                    mapped_keys.append(mapped_state)
                else:
                    mapped_probs_dict[mapped_state] += prob
            else:
                discarded_states = True

        if keep_keys:
            keys = mapped_keys

        # Convert to a PyTorch tensor while preserving autograd tracking
        probabilities = torch.stack([mapped_probs_dict[k] for k in mapped_keys], dim=1)

        # Renormalize if any states were discarded
        if discarded_states and probabilities.sum(dim=1).min() > 0:
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

    if not is_batched:
        probabilities = probabilities.squeeze(0)

    return keys, probabilities


def comb_to_state(combination: List[int], m: int) -> Tuple[int, ...]:
    return tuple([1 if i in combination else 0 for i in range(m)])


def rank_to_combination(rank: int, m: int, n: int) -> List[int]:
    combination = []
    remaining_rank = rank
    prev = -1
    for i in range(n):
        for c in range(prev + 1, m):
            available = m - c - 1
            needed = n - i - 1
            possible_combinations = math.comb(available, needed)
            if remaining_rank >= possible_combinations:
                remaining_rank -= possible_combinations
            else:
                combination.append(c)
                prev = c
                break
        else:
            raise ValueError("Invalid index or parameters")
    return combination


def combination_to_rank(
    combination: List[int],
    m: int,
    n: int,
    start: int = 0,
    previous_c: int = -1,
    rank: int = 0,
    to_cache: int = None,
) -> Tuple[int, int, int]:
    cache_previous_c = -1
    cache_rank = 0
    for i in range(start, n):
        if to_cache == i:
            cache_previous_c = previous_c
            cache_rank = rank
        c_i = combination[i]
        start = previous_c + 1
        k = n - i
        a = m - start
        b = m - c_i
        term = 0
        if a >= k:
            term += math.comb(a, k)
        if b >= k:
            term -= math.comb(b, k)
        rank += term
        previous_c = c_i
    return rank, cache_previous_c, cache_rank


def pytorch_slos_output_distribution_nobunching(
    unitary: torch.Tensor,
    input_state: list,
    output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
    keep_keys: bool = True,
) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
    """
    Optimized version of SLOS for no-bunching states only.
    Maintains the simultaneous calculation of all output amplitudes from the original SLOS
    but optimizes for binary (0/1) input and output states.

    Args:
        unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m]
        input_state (list): Binary list (0s and 1s) indicating occupied input modes
        output_map_func (callable, optional): Function that maps binary output states to new states
            or None. If the function returns None for a state, that state is discarded.
        keep_keys (bool): If True, the output state keys are returned (small overhead). Default is True.

    Returns:
        Tuple[List[Tuple[int, ...]], torch.Tensor]:
            - List of tuples representing binary output states (0s and 1s)
            - For single unitary: Probability distribution tensor [num_states]
            - For batched unitaries: Probability distributions tensor [batch_size x num_states]
    """
    if len(unitary.shape) == 2:
        is_batched = False
        unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
    else:
        is_batched = True

    batch_size, m, m2 = unitary.shape
    if m != m2:
        raise ValueError("Unitary matrices must be square")

    if len(input_state) != m:
        raise ValueError("Input state length must match unitary matrix dimension")

    if any(n < 0 for n in input_state) or sum(input_state) == 0:
        raise ValueError("Photon numbers cannot be negative or all zeros")

    if not all(x in [0, 1] for x in input_state):
        raise ValueError("Input state must be binary (0s and 1s only)")

    # Create input mode indices
    idx_n = []
    n = 0
    for i, count in enumerate(input_state):
        for _ in range(count):
            idx_n.append(i)
            n += 1

    keys = None

    for i, p in enumerate(idx_n):
        if i == 0:
            U_Fp1 = unitary[:, :, p]
        else:
            U_Fp1 = torch.zeros((batch_size, math.comb(m, i + 1)), dtype=torch.cfloat)

            for idx in range(U_F.shape[-1]):
                # insert one element that will slide to the right without complex list operation
                combination = [-1] + rank_to_combination(idx, m, i)
                pos = 1
                cache_rank = 0
                cache_previous_c = -1
                cache_start = 0
                for k in range(m):
                    if pos < i + 1 and combination[pos] == k:
                        combination[pos], combination[pos - 1] = (
                            combination[pos - 1],
                            combination[pos],
                        )
                        pos += 1
                    else:
                        combination[pos - 1] = k
                        # new_idx, _, _ = combination_to_rank(combination, m, i + 1)
                        new_idx, cache_previous_c, cache_rank = combination_to_rank(
                            combination,
                            m,
                            i + 1,
                            cache_start,
                            cache_previous_c,
                            cache_rank,
                            pos - 1,
                        )
                        cache_start = pos - 1
                        U_Fp1[:, new_idx] += unitary[:, k, p] * U_F[:, idx]

        U_F = U_Fp1

    # Calculate initial probabilities
    probabilities = (U_Fp1.abs() ** 2).real

    # Apply output mapping if provided
    if output_map_func is not None:
        # Create dictionary to accumulate probabilities for mapped states
        mapped_probs_dict = {}
        mapped_keys = []

        for idx in range(U_Fp1.shape[-1]):
            key = comb_to_state(rank_to_combination(idx, m, n), m)
            prob = probabilities[:, idx]
            mapped_state = output_map_func(key)
            if mapped_state is not None:  # Only include non-discarded states
                if mapped_state not in mapped_probs_dict:
                    mapped_probs_dict[mapped_state] = prob
                    mapped_keys.append(mapped_state)
                else:
                    mapped_probs_dict[mapped_state] += prob

        # Convert to a PyTorch tensor while preserving autograd tracking
        if keep_keys:
            keys = mapped_keys
        probabilities = torch.stack([mapped_probs_dict[k] for k in mapped_keys], dim=1)
    elif keep_keys:
        keys = [
            comb_to_state(rank_to_combination(idx, m, n), m)
            for idx in range(U_Fp1.shape[-1])
        ]

    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

    # Remove batch dimension if input was single unitary
    if not is_batched:
        probabilities = probabilities.squeeze(0)

    return keys, probabilities
