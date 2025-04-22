#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=========================================================================
#Copyright (c) June 2024

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#=========================================================================

# This work has been supported by the French Agence Nationale de la Recherche 
# through the France 2030 program under grant agreement No. ANR-22-PETQ-0008 PQ-TLS.

#=========================================================================

# Author: Clémence Chevignard, Pierre-Alain Fouque & André Schrottenloher
# Date: June 2024
# Version: 2

#=========================================================================
"""
The purpose of this script is to quickly estimate the size of quantum 
circuits given the length of an RSA modulus / DL instance.

Since we need to estimate the different parameter lengths, there is some code here that is similar to
"classical_simulation.py". But since we don't need to compute the cofactors, it
will run faster and allow to test larger parameter lengths without
having to wait for hours.

Note that these estimates rely on the values of s taken from Table 3 in 
"On post-processing in the quantum algorithm for computing short discrete logarithms" (Ekera, 2019).

The estimate for the number of qubits also relies on a heuristic. In the internal 
"multi-bit-sum" circuit, the number of ancillas required depends on the number of elements
in the sum. Assuming that the numbers in the multi-sum circuit are uniformly distributed,
when summing l numbers, we expect the multi-bit sum circuits to span only around l/2 bits.
The bound that we use is 0.65*l, which should be valid for all sub-circuits with high
probability thanks to Chernoff bounds.

"""

from math import log, ceil, floor, sqrt
import random
from quantumthreattracker.algorithms.rsa.chevignard.util import *
from qualtran.resource_counting import GateCounts as gc
from qualtran.surface_code import AlgorithmSummary
from sympy import randprime, primerange, mod_inverse
from quantumthreattracker.algorithms.rsa.chevignard.modular_multi_product import ControlledModularMultiProduct

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


def find_rns_base(u, min_bit_size=18):
    """Computes the RNS primes (we don't use 2)."""
    l = list(primerange(2**min_bit_size, 3 * u))
    final_l = []
    final_prod = 1
    for p in l:  # not 2
        final_l.append(p)
        final_prod += log(p, 2)
        if final_prod > u:
            return final_l
    raise Exception("shouldn't stop here")


def estimate_success_prob(nb_runs):
    """
    Assuming that the Ekera-Hastad post-processing succeeds with
    probability 0.99 (true for the parameters in our tables), returns
    a probability of failure for the circuit that is sufficiently low to bring
    the total probability of success of the algorithm to 0.9. 
    (See Section 5 in the paper).
    
    That is, p should satisfy 8 sqrt(nb_runs) sqrt(p) < 0.08
    
    More precisely, the function returns the number of bits of padding
    for the truncation in the computation of Q_N, and this number corresponds
    to minus the log_2 of p.
    """
    return ceil(-log((0.08 / (8 * sqrt(nb_runs)))**2, 2))


def estimate_parameters(n, flag="rsa"):
    """
    Estimates all the parameters that are necessary to estimate qubit and
    gate counts for a given instance of the algorithm.
    
    Args:
        n: bit-size of the instance (see RSA_TABLE and DLOG_TABLE)
        flag: either "rsa" or "dlog"
    """
    if flag not in ["rsa", "dlog"]:
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

    modN_success_prob_bits = estimate_success_prob(max_runs)
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
    _d = {}
    for r in rns_base:
        _bit_size = ceil(log(r, 2))
        if _bit_size not in _d:
            _d[_bit_size] = 0
        _d[_bit_size] += 1
    print("Distribution of bit-size:", _d)

    # setting the size of q_M
    alpha = ceil(log(sum(rns_base), 2))
    q_M_bit_length = max_rns_bit_size + alpha

    # choosing parameter u
    _tmp = 0
    for p in rns_base:
        _tmp += p**2
    u = ceil(log(_tmp, 2)) + 1

    # setting parameter u'
    bits_in_sum = 0
    for p in rns_base:
        bits_in_sum += ceil(log(p, 2))
    u_N = ceil(log(bits_in_sum + q_M_bit_length, 2)) + modN_success_prob_bits

    constants["nu"] = modN_success_prob_bits
    constants["max_runs"] = max_runs
    constants["average_runs"] = average_runs
    constants["n"] = n
    constants["modN_success_prob_bits"] = modN_success_prob_bits
    constants["rns_len"] = rns_len
    constants["rns_distr"] = _d
    constants["rns_max_bit_size"] = max([v for v in _d])
    constants["q_M_bit_length"] = q_M_bit_length
    constants["u"] = u
    constants["u_N"] = u_N
    constants["r"] = 22

    return constants


def average_multi_product_cost(prime_bit_size, m, trials=20):
    """Computes various random instances of the controlled modular
    multi-product circuit (computing an RNS residue) to average their
    gate counts and depth.
    """

    sum_gc = GateCounts()
    sum_depth = 0
    max_qubits = 0

    bit_size = 21
    for ctr in range(trials):

        p = randprime(2**(bit_size - 1), 2**bit_size)
        l = [random.randrange(0, p) for _ in range(m)]
        qc = ControlledModularMultiProduct(l, p, half=True)
        #qc.test()
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


def all_parameters_table(d, flag="rsa"):
    """Outputs a table (lateX style) with relevant parameters for the circuit,
    for the RSA moduli / prime bit-size given in argument. This table does not contain
    the circuit cost.
    
    Parameters:
        d - list of bit sizes
        flag - either "rsa" or "dlog"
    """

    top_line = "RSA bit-size" if flag == "rsa" else "$\\ceil{\\log_2 p}, d$"
    line_s_ell = ("""$s, \\ell = \\ceil{\\frac{n}{2s}}$""" if flag == "rsa"
                  else """$s, \\ell = \\ceil{\\frac{d}{s}}$""")
    line_m = ("""$m = \\frac{n}{2} + 2\\ell$"""
              if flag == "rsa" else "$m = d + 2\\ell$")
    line_mes = """Measurements: Eker{\\aa}, us (average)"""
    line_nbr_primes = """Number of primes in the RNS"""
    line_bit_size = """\\begin{tabular}{c}Maximal bit-size of \\\\ primes in the RNS\\end{tabular}"""
    line_u = """$u$"""
    line_up = """$u'$"""
    line_nu = """$\\nu$"""
    line_qm = """$\\ceil{\\log_2 q_M}$"""
    line_size_qm = """\\begin{tabular}{c} Size of $q_M$ register \\\\ $= \\ceil{\\log_2 q_M} + u + 1$\\end{tabular}"""
    line_size_qn = """Size of $Q_N$ register $= u' + r$"""
    line_size_res = """Size of result register $=r$"""

    for n in d:
        params = estimate_parameters(n, flag=flag)
        top_line += ("&" + str(n) if flag == "rsa" else "&" + str(n) + ", " +
                     str(params["d"]))
        line_s_ell += ("&" + str(params["s"]) + ", " + str(params["ell"]))
        line_m += "&" + str(params["m"])
        line_mes += "&" + str(params["average_runs"] // 2) + ", " + str(
            params["average_runs"])
        line_nu += "&" + str(params["nu"])
        line_nbr_primes += "&" + "\\num{" + str(params["rns_len"]) + "}"
        line_bit_size += "&" + str(params["rns_max_bit_size"])
        line_u += "&" + str(params["u"])
        line_up += "&" + str(params["u_N"])
        line_qm += "&" + str(params["q_M_bit_length"])
        line_size_qm += "&" + str((params["u"] + params["q_M_bit_length"] + 1))
        line_size_qn += "&" + str((params["u_N"] + params["r"]))
        line_size_res += "&" + str(params["r"])

    for l in [
            top_line, line_s_ell, line_m, line_mes, line_nu, line_nbr_primes,
            line_bit_size, line_u, line_up, line_qm, line_size_qm,
            line_size_qn, line_size_res
    ]:
        print(l + "\\\\")


def full_circuit_costs(n, trials=2, latex_display=False, flag="rsa"):
    """Estimates the cost of the exponentiation circuit based on the largest prime size
    and the typical cost for primes of this size, which is estimated by
    taking the average of random "modular multi-product" circuits for primes
    of this size.
    
    The result is likely to be slightly overestimated (since we focus on the largest
    prime size).
    
    Parameters:
        n - bit-size of instance
        flag - "rsa" for RSA instance or "dlog" for dlog instance or "dlog-shor" for dlog
                instance in which we use Shor's algorithm (larger input register)
    """

    params = estimate_parameters(n, flag=flag)

    m = params["m"]
    largest_prime_bit_size = params["rns_max_bit_size"]
    nbr_primes = log(params["rns_len"], 2)
    nbr_mtp_circuits = nbr_primes + 2  # number of multi-product circuits: 4 times
    # the number of primes in the RNS
    runs = log(params["average_runs"], 2)

    prettyprint = lambda x: "$2^{%s}$" % str(round(x, 2))
    scientific_num = lambda x: int(2**x)

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

    total_logical_gates = gc(toffoli=toffoli, clifford=cnot + x, measurement=depth)
    print(total_logical_gates)
    return AlgorithmSummary(
        n_algo_qubits=total_qubits,
        n_logical_gates=total_logical_gates)

    # if latex_display:
    #     # print this on a line that can be directly included in a table
    #     # Qubits (incl ancilla), Toffoli (single run) CNOT, X, Depth || Toffoli (total), etc

    #     print(
    #         "n &  & Qubits (incl. ancilla) & Toffoli & CNOT & X & Depth \\\\")
    #     print("""\multirow{3}*{%s} """ % str(n), "& Single multi-product &",
    #           multi_product_ancillas + m, "(", multi_product_ancillas, ")",
    #           "&", prettyprint(gate_counts["ccx"]), "&",
    #           prettyprint(gate_counts["cx"]), "&",
    #           prettyprint(gate_counts["x"]), "&", prettyprint(depth), "\\\\")
    #     print("& Full circuit &", total_qubits, "(", ancilla_qubits, ")", "&",
    #           prettyprint(nbr_mtp_circuits + gate_counts["ccx"]), "&",
    #           prettyprint(nbr_mtp_circuits + gate_counts["cx"]), "&",
    #           prettyprint(nbr_mtp_circuits + gate_counts["x"]), "&",
    #           prettyprint(nbr_mtp_circuits + depth), "\\\\")
    #     print("& Full algorithm &", total_qubits, "(", ancilla_qubits,
    #           ")", "&",
    #           prettyprint(nbr_mtp_circuits + runs + gate_counts["ccx"]), "&",
    #           prettyprint(nbr_mtp_circuits + runs + gate_counts["cx"]), "&",
    #           prettyprint(nbr_mtp_circuits + runs + gate_counts["x"]), "\\\\")
    # else:
    #     print("CCX", nbr_mtp_circuits + gate_counts["ccx"])
    #     print("CX", nbr_mtp_circuits + gate_counts["cx"])
    #     print("X", nbr_mtp_circuits + gate_counts["x"])
    #     print("Depth", nbr_mtp_circuits + depth)
    #     print("Qubits", total_qubits)


if __name__ == "__main__":

    flag = "rsa"

    #all_parameters_table( [2048, 3072, 4096, 6144, 8192] , flag=flag)

    # if True:
    trials = 1

    full_circuit_costs(2048, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(3072, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(4096, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(6144, trials=trials, latex_display=True, flag=flag)
    # full_circuit_costs(8192, trials=trials, latex_display=True, flag=flag)