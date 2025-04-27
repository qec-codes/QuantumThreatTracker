#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =========================================================================
# Copyright (c) June 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================================

# This work has been supported by the French Agence Nationale de la Recherche
# through the France 2030 program under grant agreement No. ANR-22-PETQ-0008 PQ-TLS.

# =========================================================================

# Author: Clémence Chevignard, Pierre-Alain Fouque & André Schrottenloher
# Date: June 2024
# Version: 2

# =========================================================================
"""
Estimate the size of quantum circuits given the length of an RSA modulus or DL instance.

Since we need to estimate the different parameter lengths, there is some code here
that is similar to "classical_simulation.py". But since we don't need to compute the
cofactors, it will run faster and allow to test larger parameter lengths without
having to wait for hours.

Note that these estimates rely on the values of s taken from Table 3 in
"On post-processing in the quantum algorithm for computing short
discrete logarithms" (Ekera, 2019).

The estimate for the number of qubits also relies on a heuristic. In the internal
"multi-bit-sum" circuit, the number of ancillas required depends on the # of elements
in the sum. Assuming that the numbers in the multi-sum circuit are uniformly distributed
when summing l numbers, we expect these circuits to span only around l/2 bits. The bound
0.65*l is used, which should be valid with high probability thanks to Chernoff bounds.
"""

import random
from math import ceil, log, sqrt

import qualtran.resource_counting as rc
from qualtran.surface_code import AlgorithmSummary
from sympy import primerange, randprime

from quantumthreattracker.algorithms.rsa.chevignard_utils.modular_multi_product import (
    ControlledModularMultiProduct,
)
from quantumthreattracker.algorithms.rsa.chevignard_utils.util import (
    GateCounts,
    full_decompose,
    gate_counts,
)

# Table 3 in Ekera's paper. For a given RSA modulus bit-size,
# gives s and the number of measurements for 99% success
# probability in the Ekera-Hastad algorithm.
# Recall that because of May-Schlieper compression, we will increase
# this number of measurements by a factor ~ 2.
RSA_TABLE = {
    2048: (17, 20),
    3072: (21, 24),
    4096: (24, 27),
    6144: (31, 34),
    8192: (34, 37)
}

# Table 2 in Ekera's paper.  For a given size of Schnorr group,
# gives s, the number of measurements *and* the value of r
# (expected dlog size, which is the exponent in the exponentiation)
DLOG_TABLE = {
    2048: (7, 10, 224),
    3072: (8, 11, 256),
    4096: (9, 12, 304),
    6144: (10, 13, 352),
    8192: (11, 14, 400),
}


def find_rns_base(u: int, min_bit_size: int = 18) -> list:
    """Compute the RNS primes (we don't use 2).

    Args:
    ----
        u (int): The upper bound for the RNS primes.
        min_bit_size (int): The minimal bit size of the primes.

    Returns
    -------
        list: A list of primes in the RNS, such that the sum of their bit sizes
        is greater than u.

    Raises
    ------
    Exception
        If the sum of the bit sizes of the primes
            is not greater than u.
    """
    primerange_list = list(primerange(2**min_bit_size, 3 * u))
    final_l = []
    final_prod = 1
    for p in primerange_list:  # not 2
        final_l.append(p)
        final_prod += log(p, 2)
        if final_prod > u:
            return final_l
    raise Exception("shouldn't stop here")


def estimate_success_prob(nb_runs: int) -> int:
    """Compute the number of bits of padding for success.

    Assuming that the Ekera-Hastad post-processing succeeds with
    probability 0.99 (true for the parameters in our tables), this function
    returns a probability of failure for the circuit sufficiently low to bring
    the total success probability of the algorithm to 0.9. (See Section 5 in the paper.)

    That is, p should satisfy 8 * sqrt(nb_runs) * sqrt(p) < 0.08.

    More precisely, the function returns the number of bits of padding
    for the truncation in the computation of Q_N, corresponding to
    minus the log₂ of p.

    Args
    ----
        nb_runs (int): The number of runs to consider.

    Returns
    -------
        int: The number of bits of padding required.
    """
    return ceil(-log((0.08 / (8 * sqrt(nb_runs)))**2, 2))


def estimate_parameters(n: int, flag: str = "rsa")  -> dict:  # noqa: PLR0915
    """Estimate parameters for quantum circuit qubit and gate counts.

    Args
    ----
        n (int): Bit-size of the instance (see RSA_TABLE and DLOG_TABLE).
        flag (str): Either "rsa" or "dlog".

    Returns
    -------
        dict: A dictionary containing estimated parameters.

    Raises
    ------
    ValueError
        If flag is not "rsa" or "dlog".
    """
    if flag not in {"rsa", "dlog"}:
        raise ValueError("Not supported")

    # output
    constants = dict()

    if flag == "rsa":
        if n not in RSA_TABLE:
            raise ValueError("Not supported")
        s, measurements = RSA_TABLE[n]
        ell = ceil((n / 2) / s)
        print("ell=", ell)
        m = n // 2 + 2 * ell  # size of multi-product
        print("m=", m)
        constants["ell"] = ell
        constants["s"] = s
        constants["m"] = m
    elif flag == "dlog-shor":
        s, measurements, d = DLOG_TABLE[n]
        constants["d"] = d
        constants["m"] = 2 * d
        measurements = 1

    else:
        if n not in DLOG_TABLE:
            raise ValueError("Not supported")
        s, measurements, d = DLOG_TABLE[n]
        constants["d"] = d
        ell = ceil(d / s)
        print("ell=", ell)
        m = d + 2 * ell  # size of multi-product
        print("m=", m)
        constants["ell"] = ell
        constants["s"] = s
        constants["m"] = m

    max_runs = 4 * measurements
    average_runs = 2 * measurements
    # the *maximal* number of runs (estimated) is 4*measurements, and we use this to set
    # the probability of success. However the *average* number of runs is
    # 2*measurements, and this gives the time complexity of the algorithm.

    # bit-size required for the RNS: ceil(log(N,2))*(m/2 + m^(2/3))
    max_bit_size = n * ceil(m / 2 + pow(m, (2 / 3)))

    modN_success_prob_bits = estimate_success_prob(max_runs)  # noqa: N806
    # additional bits to increase success probability of the reduction mod N step
    # (the computation of the quotient in the RNS reduction step already succeeds
    # with overwhelming probability)

    min_bit_size = 18  # min bit size of primes in the RNS (for uniformity)
    rns_base = find_rns_base(max_bit_size, min_bit_size=min_bit_size)
    print("Computed RNS base")

    # maximal bit size of primes in the RNS (= w in paper)
    max_rns_bit_size = ceil(log(max(rns_base), 2))
    # number of primes in RNS
    rns_len = len(rns_base)

    print("Max represented numbers bit-size:", max_bit_size)
    print("Max bit size of primes in RNS:", max_rns_bit_size)
    print("Number of primes in RNS:", rns_len)
    d_ = {}
    for r in rns_base:
        bit_size = ceil(log(r, 2))
        if bit_size not in d_:
            d_[bit_size] = 0
        d_[bit_size] += 1
    print("Distribution of bit-size:", d_)

    # setting the size of q_M
    alpha = ceil(log(sum(rns_base), 2))
    q_M_bit_length = max_rns_bit_size + alpha  # noqa: N806

    # choosing parameter u
    tmp = 0
    for p in rns_base:
        tmp += p**2
    u = ceil(log(tmp, 2)) + 1

    # setting parameter u'
    bits_in_sum = 0
    for p in rns_base:
        bits_in_sum += ceil(log(p, 2))
    u_N = ceil(log(bits_in_sum + q_M_bit_length, 2)) + modN_success_prob_bits  # noqa: N806

    constants["nu"] = modN_success_prob_bits
    constants["max_runs"] = max_runs
    constants["average_runs"] = average_runs
    constants["n"] = n
    constants["modN_success_prob_bits"] = modN_success_prob_bits
    constants["rns_len"] = rns_len
    constants["rns_distr"] = d_
    constants["rns_max_bit_size"] = max([v for v in d_])
    constants["q_M_bit_length"] = q_M_bit_length
    constants["u"] = u
    constants["u_N"] = u_N
    constants["r"] = 22

    return constants


def average_multi_product_cost(prime_bit_size: int, m: int, trials: int = 20) -> tuple:
    """Compute random instances of the ctrl modular multi-product circuit.

    Returns
    -------
        tuple: A tuple containing the average gate counts, depth, and number of ancillas
    """
    sum_gc = GateCounts()
    sum_depth = 0
    max_qubits = 0

    bit_size = 21
    for ctr in range(trials):

        p = randprime(2**(bit_size - 1), 2**bit_size)
        p_range_list = [random.randrange(0, p) for _ in range(m)]
        qc = ControlledModularMultiProduct(p_range_list, p, half=True)
        qc = full_decompose(qc, do_not_decompose=[])

        d = gate_counts(qc)
        sum_gc += d
        sum_depth += qc.depth()
        max_qubits = max(max_qubits, len(qc.qubits))

        print(ctr, "gates", {k: log(sum_gc[k] / (ctr + 1), 2) for k in sum_gc})
        print(ctr, "depth", log(sum_depth / (ctr + 1), 2))
        print(ctr, "max_qubits", max_qubits, max_qubits - m)

    return ({k: log(sum_gc[k] / trials, 2)
             for k in sum_gc}, log(sum_depth / trials, 2), max_qubits - m)


def full_circuit_costs(n: int, trials: int = 2,
                       latex_display: bool = False, flag: str = "rsa") -> AlgorithmSummary:
    """Calculate algorithm summary for full factoring circuit.

    Parameters
    ----------
        n - bit-size of instance
        flag - "rsa" for RSA instance or "dlog" for dlog instance or "dlog-shor"
            for dlog instance in which we use Shor's algorithm (larger input register)

    Returns
    -------
        AlgorithmSummary: A summary of the algorithm's resource requirements.
    """
    params = estimate_parameters(n, flag=flag)

    m = params["m"]
    largest_prime_bit_size = params["rns_max_bit_size"]
    nbr_primes = log(params["rns_len"], 2)
    nbr_mtp_circuits = nbr_primes + 2  # number of multi-product circuits: 4 times
    # the number of primes in the RNS
    # runs = log(params["average_runs"], 2)

    # prettyprint = lambda x: "$2^{%s}$" % str(round(x, 2))
    scientific_num = lambda x: int(2**x)  # noqa: E731

    # estimate cost for computing a single RNS residue
    (gate_counts, depth, multi_product_ancillas) = average_multi_product_cost(
        largest_prime_bit_size, m, trials=trials)

    # total qubits for the circuit:
    ancilla_qubits = (multi_product_ancillas +
                      (params["u"] + params["q_M_bit_length"] + 1) +
                      (params["u_N"] + params["r"]) + params["r"])
    total_qubits = ancilla_qubits + m

    toffoli = scientific_num(nbr_mtp_circuits + gate_counts["ccx"])
    cnot = scientific_num(nbr_mtp_circuits + gate_counts["cx"])
    x = scientific_num(nbr_mtp_circuits + gate_counts["x"])
    depth = scientific_num(nbr_mtp_circuits + depth)

    total_logical_gates = rc.GateCounts(toffoli=toffoli, clifford=cnot + x, measurement=depth)
    print(total_logical_gates)
    return AlgorithmSummary(
        n_algo_qubits=total_qubits,
        n_logical_gates=total_logical_gates)


if __name__ == "__main__":

    flag = "rsa"
    # if True:
    trials = 1

    full_circuit_costs(2048, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(3072, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(4096, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(6144, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(8192, trials=trials, latex_display=True, flag=flag)
