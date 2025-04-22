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

This file implements a modular multi-product circuit that we used to benchmark
our algorithm.
"""

from .util import *
from math import log, ceil, floor
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, CircuitInstruction
import numpy as np
import random
from sympy import isprime
from sympy import randprime, primerange, mod_inverse, gcd
from sympy.ntheory import discrete_log
from sympy import divisors
from sympy import factorint
from quantumthreattracker.algorithms.rsa.chevignard.basic_arithmetic import *

# =====================================
# controlled 4-bit incrementor

# there is an elegant general description for a controlled incrementor circuit:
# on bits 0,1,2,3 where 0 is the control, apply multi-controlled on 0,1,2,3 with target 3,
# then on 0,1,2 with target 2, then on 0,1 with target 1, end. More generally we
# can decompose the circuit using O(n) multi-controlled Toffolis. An ancilla is required
# to decompose the MCTs into Toffolis.

# This circuit was optimized using SAT solving. The first qubit is control,
# the last is ancilla
_qc = QuantumCircuit(6)
_qc.ccx(0, 1, 5)
_qc.ccx(2, 3, 1)
_qc.ccx(1, 5, 4)
_qc.ccx(2, 3, 1)
_qc.cx(5, 4)
_qc.ccx(2, 5, 3)
_qc.cx(5, 2)
_qc.ccx(0, 1, 5)
_qc.cx(0, 1)

CONTROLLED_INCREMENTOR_4 = _qc


def test_controlled_incrementor():
    for i in range(16):
        for b in range(2):
            input_bits = [b] + int_to_bits(i, width=4) + [0]
            if i + b >= 16:  # not supported by the circuit
                continue
            expected_output = [b] + int_to_bits(i + b, width=4) + [0]
            assert simulate(CONTROLLED_INCREMENTOR_4,
                            input_bits) == expected_output


def addition_sequence(nb_inputs):
    """
    Determine the structure of a tree of additions when we start from a given
    number of inputs.
    """
    current_regs = [i for i in range(nb_inputs)]
    res = []
    current_level = 0
    while len(current_regs) > 1:
        new_regs = []
        current_level += 1
        for i in range(len(current_regs) // 2):
            res.append((current_regs[2 * i], current_regs[2 * i + 1],
                        current_regs[2 * i], current_level))
            new_regs.append(current_regs[2 * i])
        if len(current_regs) % 2 == 1:
            new_regs.append(current_regs[-1])
        current_regs = new_regs
    return res


class MultiBitSum(QuantumCircuit):
    """Counts the number of ones in a set of input bits (Hamming weight). This is done by controlled
    incrementors for groups of bits, followed by a tree of adders with increasing
    sizes.

    Caution: this is only a half circuit. It does not put its ancillas back to 0 and
    it may even modify the input bits.

    Since we hard-coded the group size of incrementors, the ancilla count increases 
    linearly for this circuit.
    """

    def ancilla_count(nb_controls):
        """A good upper bound of the number of ancilla qubits."""
        return max(ceil(0.4 * nb_controls) - 6 + 4, 0)

    def __init__(self, nb_controls, output_size=None):
        super().__init__(name="multi_bit_sum")
        self.nb_controls = nb_controls
        self.output_size = output_size if output_size is not None else ceil(
            log(nb_controls, 2) + 1)
        input_size = nb_controls

        incrementor_size = 4
        incrementor_group_size = 2**incrementor_size - 1

        # size of initial groups: 15 bits at most to not overflow the incrementors
        nbr_groups = ceil(input_size / incrementor_group_size)
        incrementor_inputs = ([
            QuantumRegister(incrementor_group_size)
            for i in range(nbr_groups - 1)
        ] + [
            QuantumRegister(input_size -
                            (nbr_groups - 1) * incrementor_group_size)
        ])
        incrementor_outputs = [
            QuantumRegister(incrementor_size) for i in range(nbr_groups)
        ]

        addition_tree = addition_sequence(nbr_groups)
        # the result will be in register 0. For each register, the number of ancillas
        # required is 1 + the maximal level in the tree that this register reaches
        number_of_ancillas = [1 for _ in range(nbr_groups)]
        tree_height = 0
        for a, b, c, d in addition_tree:
            number_of_ancillas[a] = max(number_of_ancillas[a], d)
            number_of_ancillas[b] = max(number_of_ancillas[b], d)
            tree_height = max(tree_height, d)
        # print( number_of_ancillas )
        number_of_ancillas[0] = self.output_size - 4
        additional_ancillas = [
            QuantumRegister(number_of_ancillas[i]) for i in range(nbr_groups)
        ]
        l = sum([[incrementor_outputs[i], additional_ancillas[i]]
                 for i in range(nbr_groups)], [])

        self.add_register(*incrementor_inputs, *l)
        self.ancilla_nbr = len(self.qubits) - input_size - self.output_size

        # ===================

        for i in range(nbr_groups):
            nb = len(incrementor_inputs[i])
            for j in range(len(incrementor_inputs[i])):
                self.append(
                    CONTROLLED_INCREMENTOR_4, [incrementor_inputs[i][j]] +
                    incrementor_outputs[i][:] + [additional_ancillas[i][0]])

        # then tree of additions
        for a, b, c, d in addition_tree:
            # a: input register for addition
            # b: input register for addition
            # c: output register for addition = a
            # d: current level (tells us the size of the current register)
            # adder on 2*n + 2 bits (one carry and one ancilla)
            adder = HalfAdder(bit_size=incrementor_size + d - 1)
            # x y -> x x+y
            # carry and output bit should be both 0
            # ! we can remove half of the adder here, because we don't reuse x
            # after this step
            self.append(
                adder,
                incrementor_outputs[b][:] + additional_ancillas[b][:(d - 1)] +
                incrementor_outputs[a][:] + additional_ancillas[a][:(d - 1)] +
                [additional_ancillas[a][d - 1]] +
                [additional_ancillas[b][d - 1]])

    def test(self):
        for _ in range(50):
            input_bits = (
                [random.randrange(2) for i in range(self.nb_controls)] + [0] *
                (len(self.qubits) - self.nb_controls))
            expected_output = (
                input_bits[:self.nb_controls] +
                int_to_bits(sum(input_bits), width=self.output_size))
            output_bits = simulate(self, input_bits)
            assert (output_bits[self.nb_controls:(
                self.nb_controls + self.output_size)] == int_to_bits(
                    sum(input_bits), width=self.output_size))


class MultiIntegerSum(QuantumCircuit):
    """Performs a multi-controlled sum of integers. It uses several layers of
    multi-bit additions. """

    def __init__(self, l, output_size=None, max_ancillas=False):
        """
        Parameters:
            l - list of integers for the multi-sum
            output_size - required size of the output register
            max_ancillas - if set to True, will allocate the maximal number
                        of ancillas which would be required for such a case
        """
        super().__init__(name="multi_integer_sum")
        self.l = l
        levels = 0  # nbr of levels with multi-bit additions
        nbr_inputs = len(l)
        for i in l:
            if i > 0:
                levels = max(levels, ceil(log(i, 2)))

        self.output_size = output_size if output_size is not None else ceil(
            log(sum(l), 2))
        assert self.output_size >= ceil(log(sum(l), 2))

        adder_ancillas = QuantumRegister(2)
        controls = QuantumRegister(nbr_inputs)
        output_reg = QuantumRegister(self.output_size)

        sub_circuits = {}
        sub_circuits_inputs = {}
        # determine the multi-bit addition circuits for each level
        for i in range(levels):
            # list of inputs for the controls
            sub_circuits_inputs[i] = [
                j for j in range(len(l)) if ((l[j] >> i) % 2 == 1)
            ]
            # perform a multi-sum circuit, but only with this list of bits
            if sub_circuits_inputs[i] != []:
                sub_circuits[i] = MultiBitSum(len(sub_circuits_inputs[i]),
                                              max(self.output_size, 5))

        # check size of all sub_circuits: required nbr of ancillas
        nbr_ancilla = {}
        for i in sub_circuits:
            nbr_ancilla[i] = len(sub_circuits[i].qubits) - len(
                sub_circuits_inputs[i])

        self.add_register(controls, output_reg)

        if not sub_circuits:
            # empty circuit
            return

        if max_ancillas:
            if len(l) < 1000:
                # in that case, no bound
                ancilla_reg = QuantumRegister(
                    max(self.output_size, 5) + MultiBitSum.ancilla_count(len(l)))
                # raise ValueError("Unsupported: l should be large enough for the bound to be correct w.h.p")
            # when l >= 1000, the probability that a multi-bit sum has more than 1.3 * l/2
            # bits to sum is lower than exp(-0.3**2 / (2+0.3) * l/2 ) <= exp(−0.0195 * l ) <= exp(-19.6)
            # which is enough to have this satisfied for all instances of multi-bit-sum in the circuit
            ancilla_reg = QuantumRegister(
                max(self.output_size, 5) +
                MultiBitSum.ancilla_count(int(0.65 * len(l))))
        else:
            ancilla_reg = QuantumRegister(
                max([nbr_ancilla[i] for i in nbr_ancilla]))

        self.add_register(ancilla_reg, adder_ancillas)

        adder = HalfAdder(bit_size=self.output_size)

        for i in sub_circuits:
            # apply the current level
            self.append(sub_circuits[i],
                        [controls[j] for j in sub_circuits_inputs[i]] +
                        ancilla_reg[:nbr_ancilla[i]])
            # the result of the current level is stored in the first
            # "output_size" qubits of ancilla_reg
            # we add it to output reg (accumulator)
            tmp = ancilla_reg[:self.output_size]
            shifted_result = tmp[-i:] + tmp[:-i]
            self.append(adder,
                        shifted_result + output_reg[:] + adder_ancillas[:])
            # uncompute the level
            self.append(sub_circuits[i].inverse(),
                        [controls[j] for j in sub_circuits_inputs[i]] +
                        ancilla_reg[:nbr_ancilla[i]])

    def test(self):
        in_len = len(self.l)
        out_len = self.output_size
        for _ in range(20):
            input_bits = ([random.randrange(2) for i in range(in_len)] + 
                          [0] * (len(self.qubits) - in_len))
            s = sum([self.l[i] * input_bits[i] for i in range(in_len)])
            expected_output = (input_bits[:in_len] + int_to_bits(s, width=out_len) + 
                            [0] * (len(self.qubits) - in_len - out_len))
            output_bits = simulate(self, input_bits)
            assert expected_output == output_bits


def find_generator(p):
    """Finds a multiplicative generator of the group Z_p^*. """
    div = divisors(p - 1)[:-1]
    for i in range(3, p):
        good = True
        for d in div:
            if pow(i, d, p) == 1:
                good = False
                break
        if good:
            return i
    raise Exception("couldn't find a generator ??")


class ControlledModularMultiProduct(QuantumCircuit):
    """Circuit that performs a controlled multi-product modulo a small prime.
    It is optimized for a prime p of 21 bits.

    """

    def __init__(self, l, p, verb=False, half=True):
        """

        Args:
        - l: list of integers for the product (modulo p)
        - p: the prime
        - half: if True, will not uncompute the sum of discrete logarithms (this
        costs slightly more ancillas, but saves a factor ~2 in gate count).
        """

        super().__init__(name="controlled_modular_multi_product")
        self.p = p
        self.l = l
        self.g = find_generator(p)

        generator = self.g
        nbr_inputs = len(l)
        log2p = ceil(log(p, 2))
        self.log2p = log2p

        dlogs = {}
        zero = []  # some of the integers can be 0 mod p. We must remove them
        # from the product, because they have no dlog.
        for i in range(len(l)):
            if l[i] % p != 0:
                dlogs[i] = discrete_log(p, l[i], generator)
            else:
                zero.append(i)

        dlogs_sum_size = ceil(log(sum([dlogs[i] for i in dlogs]), 2))

        sum_entries_nbr = len([dlogs[i] for i in dlogs])
        # use a bound on the number of ancillas that we might need
        sum_circuit = MultiIntegerSum([dlogs[i] for i in dlogs],
                                      dlogs_sum_size,
                                      max_ancillas=True)

        _tmp = full_decompose(sum_circuit, do_not_decompose=[])
        if verb:
            print("sum circuit gates", dict(_tmp.count_ops()))
            print("sum circuit depth", _tmp.depth())
        # reduction of the sum of dlogs mod p-1 (because a^(p-1) = 1)
        euclidean_division = EuclideanDivider(p - 1, dlogs_sum_size)

        # bits of reduced sum will be the controls of the modular product!
        final_modular_product = ControlledModularProduct(
            [pow(generator, 2**i, p) for i in range(log2p)],
            p,
            controlled=True)
        _tmp = full_decompose(final_modular_product, do_not_decompose=[])
        if verb:
            print("modular product gates", dict(_tmp.count_ops()))
            print("modular product depth", _tmp.depth())

        ancillas_for_euclidean_division = euclidean_division.ancilla_nbr
        ancillas_for_sum = len(
            sum_circuit.qubits) - sum_entries_nbr - dlogs_sum_size

        if len(zero) >= 1:
            mct = MCX(len(zero))
            ancillas_for_mct = len(mct.qubits) - len(zero) - 1
        else:
            mct = None
            ancillas_for_mct = 0
        ancillas_for_mod_product = len(
            final_modular_product.qubits) - 2 * log2p
        if verb:
            print("ancillas for sum", ancillas_for_sum)
            print("ancillas for mod product", ancillas_for_mod_product)

        ancilla_count = max(ancillas_for_euclidean_division, ancillas_for_sum,
                            ancillas_for_mct, ancillas_for_mod_product)

        controls = QuantumRegister(nbr_inputs)
        output_reg = QuantumRegister(log2p)
        quotient_pad = QuantumRegister(2)

        dlog_sum = QuantumRegister(dlogs_sum_size)
        result_is_not_zero = QuantumRegister(1)
        ancillas = QuantumRegister(ancilla_count)

        self.ancilla_nbr = 1 + len(dlog_sum) + len(quotient_pad) + len(
            ancillas)
        self.garbage_nbr = 1 + len(dlog_sum) + len(quotient_pad)

        self.add_register(controls, output_reg, result_is_not_zero, dlog_sum,
                          quotient_pad, ancillas)

        # step 1: sum all the dlogs
        self.append(sum_circuit, [controls[i] for i in dlogs] + dlog_sum[:] +
                    ancillas[:ancillas_for_sum])

        # step 2: reduce the dlog output modulo p-1
        self.append(
            euclidean_division, dlog_sum[:] + quotient_pad[:] +
            ancillas[:euclidean_division.ancilla_nbr])
        # now the first bits of dlog_sum[:] contain the bits of exponent

        # step 3: if one of the control bits in "zero" is 1, then result is 0:
        # write output_reg[0] only if all of them are 0
        for i in zero:
            self.x(controls[i])
        if len(zero) == 0:
            self.x(result_is_not_zero)
        else:
            self.append(mct, [controls[i] for i in zero] +
                        result_is_not_zero[:] + ancillas[:ancillas_for_mct])
        for i in zero:
            self.x(controls[i])

        # if result_is_not_zero contains 1, then the modular multi-product
        # will return a nonzero result
        # step 4:

        self.append(
            final_modular_product, result_is_not_zero[:] + dlog_sum[:log2p] +
            output_reg[:] + ancillas[:final_modular_product.ancilla_nbr])

        # output register contains the wanted result
        for i in zero:
            self.x(controls[i])
        if len(zero) == 0:
            self.x(result_is_not_zero)
        else:
            self.append(mct, [controls[i] for i in zero] +
                        result_is_not_zero[:] + ancillas[:ancillas_for_mct])
        for i in zero:
            self.x(controls[i])

        # in "half" mode, dlog_sum[:] is not uncomputed
        if not half:
            # uncompute
            self.append(
                euclidean_division.inverse(), dlog_sum[:] + quotient_pad[:] +
                ancillas[:euclidean_division.ancilla_nbr])
            self.append(sum_circuit.inverse(), [controls[i] for i in dlogs] +
                        dlog_sum[:] + ancillas[:ancillas_for_sum])

    def test(self):
        nbr_inputs = len(self.l)
        log2p = self.log2p

        for k in range(5):
            print("testing", k)
            input_bits = [random.randrange(2) for _ in range(nbr_inputs)
                          ] + [0] * log2p + [0] * self.ancilla_nbr
            expected_nbr = 1
            for i in range(nbr_inputs):
                if input_bits[i]:
                    expected_nbr = (expected_nbr * self.l[i]) % self.p

            output_bits = simulate(self, input_bits)
            expected_output = (input_bits[:nbr_inputs] +
                               int_to_bits(expected_nbr, width=log2p) +
                               [0] * self.ancilla_nbr)
            assert (output_bits[:nbr_inputs] == input_bits[:nbr_inputs])
            assert (
                output_bits[-(self.ancilla_nbr - self.garbage_nbr):] == [0] *
                (self.ancilla_nbr - self.garbage_nbr))
            assert (output_bits[nbr_inputs:(nbr_inputs +
                                            log2p)] == int_to_bits(
                                                expected_nbr, width=log2p))


if __name__ == "__main__":

    def test_multibit_sum():
        n = 1000
        qc = MultiBitSum(n, output_size=40)
        qc = full_decompose(qc, do_not_decompose=[])
        q = gate_counts(qc)
        print(q)
        print(qc.depth())
        print(len(qc.qubits) - n - 40)
        print(0.4 * n - 6)

    def test_multi_product():
        """Computes various random instances of the circuit to average their
        gate counts and depth.
        """

        bit_size = 22
        p = randprime(2**(bit_size - 1), 2**bit_size)
        l = [random.randrange(0, p) for _ in range(1146)]
        qc = ControlledModularMultiProduct(l, p, half=True)
        qc.test()
        qc = full_decompose(qc, do_not_decompose=[])
        d = gate_counts(qc)
        print(d)
        print(qc.depth())
        print(len(qc.qubits) - 1146)

    test_multi_product()
