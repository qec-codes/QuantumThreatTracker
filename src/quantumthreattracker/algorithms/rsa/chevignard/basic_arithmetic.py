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
Implements basic arithmetic components as QuantumCircuits.
Since our implementation is fully classical, we use only CNOT, Toffoli and X (NOT)
gates.
"""

from quantumthreattracker.algorithms.rsa.chevignard.util import *
from math import log, ceil, floor
from qiskit import QuantumCircuit, QuantumRegister
import random
from sympy import randprime


class HalfAdder(QuantumCircuit):
    """
    Adder without incoming carry. Simplified version of the CDKM ripple-carry adder
    which was initially taken from Qiskit, then modified to allow various new
    cases:
    
    - "controlled": control becomes the first qubit. Last 2 qubits are carry and ancilla.
    - "comparator": we only compute the last carry bit, the rest is uncomputed.
        Then we will use the bit-complement trick to transform this into a comparator.
    - "comparator_adder": the addition is done, but only if it overflows. The
        bit of carry remembers whether this occurred. This is useful for implementing
        a Euclidean division.
    - "modular": does not compute the carry bit. In that case the circuit has
        1 less qubit.
    
    x y -> x  y+x
    """

    def __init__(self,
                 bit_size,
                 controlled=False,
                 special=None,
                 modular=False):
        """Creates the addition circuit.
        
        Args:
        - bit-size: size of the numbers to add, in bits
        - controlled: Boolean determining if the circuit is controlled
        - special: flag that can be either "comparator_adder" or "comparator"
        - modular: Boolean determining if the circuit is modular
        
        Some combinations of these inputs are possible, but not all:
        - a circuit cannot be modular and comparator
        - a circuit cannot be controlled and comparator_adder
        
        """

        if modular and special in ["comparator_adder", "comparator"]:
            raise ValueError("Incompatible parameters")
        if controlled and special == "comparator_adder":
            raise ValueError("Incompatible parameters")
        name = "half_adder_" + str(bit_size)
        if controlled:
            name += "_controlled"
        if special is not None:
            name += ("_" + special)

        super().__init__(name="half_adder_" + str(bit_size))
        self.bit_size = bit_size
        self.modular = modular
        self.controlled = controlled

        self.special = special
        self.ancilla_nbr = 2 if not modular else 1

        if self.controlled:
            control = QuantumRegister(1)
            self.add_register(control)

        qr_x = QuantumRegister(bit_size)
        qr_y = QuantumRegister(bit_size)
        ancilla = QuantumRegister(1)
        self.add_register(qr_x)
        self.add_register(qr_y)
        if not modular:
            carry = QuantumRegister(1)
            self.add_register(carry)
        self.add_register(ancilla)

        # gate for majority
        _qc = QuantumCircuit(3, name="MAJ")
        _qc.cx(0, 1)
        _qc.cx(0, 2)
        _qc.ccx(2, 1, 0)
        maj_gate = _qc.to_gate()

        if self.controlled or special == "comparator_adder":
            _qc = QuantumCircuit(4, name="UMA")
            _qc.ccx(2, 1, 0)
            _qc.cx(0, 2)
            # if control is 1: apply UMA gate
            _qc.ccx(3, 2, 1)
            # if control is 0: apply inverse of MAJ
            _qc.ccx(3, 0, 1)
            _qc.cx(0, 1)
        elif special == "comparator":
            _qc = QuantumCircuit(3, name="UMA")
            # always apply inverse of MAJ
            _qc.ccx(2, 1, 0)
            _qc.cx(0, 1)
            _qc.cx(0, 2)
        else:
            _qc = QuantumCircuit(3, name="UMA")
            # always apply the normal UMA gate
            _qc.ccx(2, 1, 0)
            _qc.cx(0, 2)
            _qc.cx(2, 1)
        uma_gate = _qc.to_gate()

        self.append(maj_gate, [qr_x[0], qr_y[0], ancilla])
        for i in range(bit_size - 1):
            self.append(maj_gate, [qr_x[i + 1], qr_y[i + 1], qr_x[i]])

        # write the carry
        if not self.modular:
            if self.controlled:
                self.ccx(control, qr_x[-1], carry)
            else:
                self.cx(qr_x[-1], carry)

        for i in reversed(range(bit_size - 1)):
            if self.controlled:
                self.append(uma_gate,
                            [qr_x[i + 1], qr_y[i + 1], qr_x[i], control])
            elif special == "comparator_adder":
                self.append(uma_gate,
                            [qr_x[i + 1], qr_y[i + 1], qr_x[i], carry])
            else:
                self.append(uma_gate, [qr_x[i + 1], qr_y[i + 1], qr_x[i]])

        if self.controlled:
            self.append(uma_gate, [qr_x[0], qr_y[0], ancilla, control])
        elif special == "comparator_adder":
            self.append(uma_gate, [qr_x[0], qr_y[0], ancilla, carry])
        else:
            self.append(uma_gate, [qr_x[0], qr_y[0], ancilla])

    def test(self):
        for _ in range(20):
            x = random.randrange(1 << self.bit_size)
            y = random.randrange(1 << self.bit_size)
            if self.controlled:
                control = [random.randrange(2)]
            else:
                control = []
            if not self.modular:
                padding = [0, 0]
            else:
                padding = [0]
            input_bits = (control + int_to_bits(x, width=self.bit_size) +
                          int_to_bits(y, width=self.bit_size) + padding)
            output_bits = simulate(self, input_bits)

            if self.controlled and control[0] == 0:
                expected_output = input_bits
            elif self.special == "comparator_adder":
                # add x to y but only if the carry is 1 (and keep the carry in all cases)
                if x + y >= (1 << self.bit_size):
                    expected_output = (
                        int_to_bits(x, width=self.bit_size) +
                        int_to_bits(y + x, width=self.bit_size + 1) + [0])
                else:
                    expected_output = (int_to_bits(x, width=self.bit_size) +
                                       int_to_bits(y, width=self.bit_size) +
                                       [0, 0])
            elif self.special == "comparator":
                expected_output = (control +
                                   int_to_bits(x, width=self.bit_size) +
                                   int_to_bits(y, width=self.bit_size) +
                                   [(x <= y)] + [0])
            elif self.modular:
                expected_output = (
                    control + int_to_bits(x, width=self.bit_size) +
                    int_to_bits(
                        (y + x) %
                        (1 << self.bit_size), width=self.bit_size) + [0])
            else:
                expected_output = (
                    control + int_to_bits(x, width=self.bit_size) +
                    int_to_bits(y + x, width=self.bit_size + 1) + [0])

            assert output_bits == expected_output
        print(self.name, "test passed")


class MCX(QuantumCircuit):
    """Implementation of a multi-controlled Toffoli gate, using clean ancillas.
    This is a standard implementation, not optimal in space. It was initially
    taken from:
    https://quantumcomputing.stackexchange.com/questions/35119/
        questions-on-multi-controlled-toffolis-and-their-implementation-in-qiskit
    """

    def __init__(self, bit_size):
        """Creates the MCX circuit.
        
        Args:
        - bit_size: number of input bits
        """
        super().__init__(name="mcx_" + str(bit_size))
        self.bit_size = bit_size

        qr_c = QuantumRegister(bit_size)
        qr_t = QuantumRegister(1)
        self.ancilla_nbr = max(bit_size - 2, 0)

        self.add_register(qr_c, qr_t)
        if bit_size > 2:
            qr_anc = QuantumRegister(bit_size - 2)
            self.add_register(qr_anc)

        if bit_size == 1:
            self.cx(0, 1)
        elif bit_size == 2:
            self.ccx(qr_c[0], qr_c[1], qr_t[0])
        else:
            self.ccx(qr_c[0], qr_c[1], qr_anc[0])
            for i in range(bit_size - 3):
                self.ccx(qr_anc[i], qr_c[i + 2], qr_anc[i + 1])
            self.ccx(qr_anc[bit_size - 3], qr_c[bit_size - 1], qr_t[0])
            for i in range(bit_size - 4, -1, -1):
                self.ccx(qr_anc[i], qr_c[i + 2], qr_anc[i + 1])
            self.ccx(qr_c[0], qr_c[1], qr_anc[0])

    def test(self):
        for _ in range(20):
            controls = [random.randrange(2) for _ in range(self.bit_size)]
            input_bits = (controls + [random.randrange(2)] +
                          [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = input_bits
            expected_output[len(controls)] ^= all(controls)
            assert output_bits == expected_output
        print(self.name, "test passed")


class ConstantMultiplier(QuantumCircuit):
    """A quantum circuit that multiplies an input by a constant.
    """

    def __init__(self, c, c_bit_size, input_bit_size):
        """Creates the circuit.
        
        Args:
        - c: the constant to multiply by
        - c_bit_size: the bit-size of c. This parameter is used to determine the
            size of the output register. (It could be bigger than strictly necessary)
        - input_bit_size: bit_size of the input.
        
        """
        super().__init__(name="constant_mul")
        self.c = c
        self.c_bit_size = c_bit_size
        self.input_bit_size = input_bit_size
        self.output_bit_size = c_bit_size + input_bit_size
        self.ancilla_nbr = 1

        input_reg = QuantumRegister(input_bit_size)
        output_reg = QuantumRegister(c_bit_size + input_bit_size)
        adder_ancilla = QuantumRegister(1)

        self.add_register(input_reg, output_reg, adder_ancilla)
        adder = HalfAdder(bit_size=input_bit_size)

        bits_of_c = int_to_bits(c, width=c_bit_size)
        for i in range(self.c_bit_size):
            if bits_of_c[i] == 1:
                self.append(
                    adder,
                    input_reg[:] + output_reg[i:(i + input_bit_size + 1)] +
                    [adder_ancilla[0]])

    def test(self):
        for _ in range(20):
            x = random.randrange(1 << self.input_bit_size)
            input_bits = (int_to_bits(x, width=self.input_bit_size) + [0] *
                          (self.output_bit_size) + [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = (
                int_to_bits(x, width=self.input_bit_size) +
                int_to_bits(x * self.c, width=self.output_bit_size) +
                [0] * self.ancilla_nbr)
            assert output_bits == expected_output
        print(self.name, "test passed")


class Multiplier(QuantumCircuit):
    """A quantum circuit that multiplies two numbers (not modularly)."""

    def __init__(self, bit_size_x, bit_size_y):
        """Creates the circuit.
        
        Args:
        - bit_size_x: bit size of the first input
        - bit_size_y: bit size of the second input
        """
        name = "mul_" + str(bit_size_x) + "_" + str(bit_size_y)
        super().__init__(name=name)
        self.bit_size_x = bit_size_x
        self.bit_size_y = bit_size_y
        self.output_bit_size = bit_size_x + bit_size_y
        self.ancilla_nbr = 1

        x_reg = QuantumRegister(bit_size_x)
        y_reg = QuantumRegister(bit_size_y)
        output_reg = QuantumRegister(self.output_bit_size)
        ancilla_reg = QuantumRegister(1)
        c_adder = HalfAdder(bit_size=bit_size_y, controlled=True)
        self.add_register(x_reg, y_reg, output_reg, ancilla_reg)

        for i in range(bit_size_x):
            # controlled on bit number i of x, sum a to the output reg, but shifted
            self.append(c_adder, [x_reg[i]] + y_reg[:] +
                        output_reg[i:(bit_size_y + i + 1)] + [ancilla_reg[0]])

    def test(self):
        for _ in range(20):
            x = random.randrange(1 << self.bit_size_x)
            y = random.randrange(1 << self.bit_size_y)
            input_bits = (int_to_bits(x, width=self.bit_size_x) +
                          int_to_bits(y, width=self.bit_size_y) + [0] *
                          (self.output_bit_size) + [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = (int_to_bits(x, width=self.bit_size_x) +
                               int_to_bits(y, width=self.bit_size_y) +
                               int_to_bits(x * y, width=self.output_bit_size) +
                               [0] * self.ancilla_nbr)
            assert output_bits == expected_output
        print(self.name, "test passed")


class ConstantComparator(QuantumCircuit):
    """A quantum circuit that compares its input with a constant.
    It writes 1 in the output if x >= c (or 0 if x < c).
    """

    def __init__(self, c, bit_size):
        """Creates the circuit.
        
        Args:
        - c: the constant to compare to
        - bit_size: bit size of the input
        """
        super().__init__(name="const_comp")
        if ceil(log(c, 2)) > bit_size:
            raise ValueError("Constant too big (not implemented)")
        self.bit_size = bit_size
        self.ancilla_nbr = bit_size + 2
        self.c = c

        bits_of_c = int_to_bits(c, width=bit_size)
        bits_for_comparator = int_to_bits(
            bits_to_int([1 - b for b in bits_of_c]) + 1, width=bit_size)

        input_reg = QuantumRegister(bit_size)
        output_reg = QuantumRegister(1)
        c_reg = QuantumRegister(bit_size)
        ancilla_reg = QuantumRegister(2)

        self.add_register(input_reg, output_reg, c_reg, ancilla_reg)
        compute_carry = HalfAdder(bit_size=bit_size,
                                  controlled=False,
                                  special="comparator")

        for i in range(bit_size):
            if bits_for_comparator[i]:
                self.x(c_reg[i])
        self.append(compute_carry,
                    c_reg[:] + input_reg[:] + output_reg[:] + ancilla_reg[1:])
        for i in range(bit_size):
            if bits_for_comparator[i]:
                self.x(c_reg[i])

    def test(self):
        for _ in range(20):
            x = random.randrange(1 << self.bit_size)
            input_bits = (int_to_bits(x, width=self.bit_size) + [0] +
                          [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = (int_to_bits(x, width=self.bit_size) +
                               [int(x >= self.c)] + [0] * self.ancilla_nbr)
            assert output_bits == expected_output
        print(self.name, "test passed")


class Incrementor(QuantumCircuit):

    def __init__(self, bit_size):
        """Creates the circuit.
        """
        super().__init__(name="incrementor")
        self.bit_size = bit_size
        self.ancilla_nbr = bit_size + 1

        input_reg = QuantumRegister(bit_size)
        tmp_reg = QuantumRegister(bit_size)
        ancilla_reg = QuantumRegister(1)

        self.add_register(input_reg, tmp_reg, ancilla_reg)

        adder = HalfAdder(bit_size=bit_size, controlled=False, modular=True)
        self.x(tmp_reg[0])

        self.append(adder, tmp_reg[:] + input_reg[:] + ancilla_reg[:])
        self.x(tmp_reg[0])

    def test(self):
        for _ in range(10):
            x = random.randrange(1 << self.bit_size)
            input_bits = (int_to_bits(x, width=self.bit_size) +
                          [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = (int_to_bits(
                (x + 1) % 2**(self.bit_size), width=self.bit_size) +
                               [0] * self.ancilla_nbr)
            assert output_bits == expected_output


class EuclideanDivider(QuantumCircuit):
    """Quantum circuit that performs a Euclidean division by a constant.
    
    Given integer p and input of size bit_size, separates the input into
    quotient and remainder in the Euclidean division by p. This is done via a
    series of subtractions in place.
    
    The input register is first padded by 2 bits. The remainder will occupy 
    the first ceil(log(p,2)) bits. The quotient occupies the rest of the bits.
    """

    def __init__(self, p, bit_size):
        """Creates the circuit.
        
        Args:
        - p: the number to divide by
        - bit_size: the bit size of the input
        """
        super().__init__(name="euclidean_divider_" + str(p) + "_" +
                         str(bit_size))
        self.p = p
        self.bit_size = bit_size
        p_bit_size = ceil(log(p, 2))

        self.p_bit_size = p_bit_size
        bits_of_p = int_to_bits(p, width=p_bit_size)
        # bit size of quotient
        self.q_bit_size = self.bit_size - p_bit_size + 1

        if self.q_bit_size < 0:
            raise ValueError("not supported")

        input_reg = QuantumRegister(self.bit_size)
        # remainder will be in input_reg. Quotient can be read off the last
        # bit_size + 2 - p_bit_size)
        input_pad = QuantumRegister(2)

        padded_input_reg = input_reg[:] + input_pad[:]

        ancilla_reg = QuantumRegister(self.p_bit_size + 2)
        self.ancilla_nbr = len(ancilla_reg)

        self.add_register(input_reg, input_pad, ancilla_reg)

        bits_for_comparator = int_to_bits(
            bits_to_int([1 - b for b in bits_of_p]) + 1, width=p_bit_size + 1)
        bits_for_comparator[-1] = 1

        comparator_adder = HalfAdder(bit_size=p_bit_size + 1,
                                     special="comparator_adder")

        for j in range(p_bit_size + 1):
            if bits_for_comparator[j]:
                self.x(ancilla_reg[j])

        for i in reversed(range(self.q_bit_size)):
            self.append(
                comparator_adder, ancilla_reg[:(p_bit_size + 1)] +
                (padded_input_reg)[i:(i + p_bit_size + 1)] +
                [padded_input_reg[-(self.q_bit_size - i)]] + ancilla_reg[-1:])

        for j in range(p_bit_size + 1):
            if bits_for_comparator[j]:
                self.x(ancilla_reg[j])
        # move middle "0" bit to end
        self.swap(p_bit_size, len(self.qubits) - 1)

    def test(self):
        for _ in range(20):
            x = random.randrange(1 << self.bit_size)
            input_bits = (int_to_bits(x, width=self.bit_size) + [0] * 2 +
                          [0] * self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = int_to_bits(x % self.p, width=self.p_bit_size)
            for i in range(len(expected_output)):
                assert output_bits[i] == expected_output[i]

            expected_output = (
                int_to_bits(x % self.p, width=self.p_bit_size + 1) +
                int_to_bits(x // self.p,
                            width=self.bit_size - self.p_bit_size + 1) +
                [0] * self.ancilla_nbr)
            #print(output_bits)
            #print(expected_output)
            assert output_bits == expected_output
        print(self.name, "test passed")


#============================================


class TableLookup(QuantumCircuit):
    """Quantum circuit that performs a lookup of a table in superposition.
    I.e., given a classical table T, on input x, it writes T[x] in the output.
    
    The approach that we implement here is the one of:
    'Encoding electronic spectra in quantum circuits with linear T complexity'
    (Babbush, Gidney, Berry, Wiebe, McClean, Paler, Fowler, Neven, Physical Review X, 2018)
    https://arxiv.org/pdf/1805.03662.pdf, 
    """

    def __init__(self, d, bit_size):
        """Creates the circuit.
        
        Args:
        - d: a dictionary that contains the table. The keys must be 0-1 tuples which
            indicate the corresponding input. The values must be integers. The number
            of keys determines the size of the input register.
        - bit_size: the bit size of the *output*
        """
        super().__init__(name="table_lookup" + str(hash(str(d))))

        self.nb_controls = int(log(len([v for v in d]), 2))
        if self.nb_controls < 2:
            raise ValueError("Unsupported")

        # we need as many ancillas as controls. If x0, x1, ... x(n-1)  are the controls, then
        # these ancillas will store the "products": u0 = (x0 + b0), u1 = (x0+b0)(x1 + b1), u2 = ...
        # where b0, b1 ... are the (opposite of the) bits of the current index in the lookup.
        # During the loop, the control bits are negated (in place) to match the current index.
        # Then, the "products" are updated. The idea is that we only need to update a few
        # of them at each loop, depending on the position at which the sequence of bi has changed.

        self.d = d
        self.ancilla_nbr = self.nb_controls - 1
        self.bit_size = bit_size

        controls = QuantumRegister(self.nb_controls)
        ancilla = QuantumRegister(self.ancilla_nbr)
        products = [controls[0]
                    ] + ancilla[:]  # makes it easier to write the computations

        output_reg = QuantumRegister(self.bit_size)
        self.add_register(controls, output_reg, ancilla)

        # ======
        # the following is an optimization of how we perform the CNOTs for data loading.
        # Notice that if two bits in successive data entries have the same value, then we can CNOT a
        # single time and not twice, and this will not depend on the last Toffoli "product",
        # but the second to last one.

        elts = dict()
        for i in range(1 << self.nb_controls):
            ind = list(reversed(int_to_bits(i, width=self.nb_controls)))
            complement_ind = [1 - t for t in ind]
            v = int_to_bits(d[tuple(complement_ind)], width=bit_size)
            elts[i] = v

        # We will use these dictionaries to indicate if we need to perform a CNOT for this
        # i, and at which level.
        do_cnot = {}
        for k in range(1, self.nb_controls):
            do_cnot[k] = {
                pos: {i: 0
                      for i in range(1 << self.nb_controls)}
                for pos in range(self.bit_size)
            }
            # do_cnot[k][pos][i] <=> must perform self.cx( products[-k], output_reg[pos] ) at index i

        for pos in range(self.bit_size):
            for i in range(1 << self.nb_controls):
                if elts[i][pos] == 1:
                    do_cnot[1][pos][i] = 1

            for k in range(2, self.nb_controls):
                for i in range(1 << self.nb_controls):
                    if i % (2**(k - 1)) == 0 and all(
                        [elts[j][pos] for j in range(i, i + 2**(k - 1))]):
                        do_cnot[k][pos][i] = 1
                        for j in range(i, i + 2**(k - 1)):
                            for kk in range(1, k):
                                # erase from lower-level dictionaries
                                do_cnot[kk][pos][j] = 0
        #=================

        # initialize the ancillas
        self.ccx(controls[0], controls[1], ancilla[0])
        for j in range(self.nb_controls - 2):
            self.ccx(controls[j + 2], ancilla[j], ancilla[j + 1])

        for i in range(1 << self.nb_controls):
            ind = list(reversed(int_to_bits(i, width=self.nb_controls)))
            complement_ind = [1 - t for t in ind]

            # the trick here is that we only need to compute onwards from the
            # first position which differs from the previous index
            # (which will be a 0 flipped to a 1)
            lowest_different_bit = None
            if i > 0:
                prev_ind = list(
                    reversed(int_to_bits(i - 1, width=self.nb_controls)))
                lowest_different_bit = 0
                for k in reversed(range(self.nb_controls)):
                    if prev_ind[k] != ind[k]:
                        lowest_different_bit = k

                # erase ancillas
                # if j := lowest_different_bit then we erase from j+1 onwards (with current controls)
                # then we CNOT j-1 on j
                # then we rewrite from j+1 onwards (with new controls)

                # this would be the naive version
                #for j in reversed(range(1, self.nb_controls)):
                #    self.ccx( controls[j], products[j-1], products[j])

                for j in reversed(
                        range(lowest_different_bit + 1, self.nb_controls)):
                    self.ccx(controls[j], products[j - 1], products[j])
                if lowest_different_bit > 0:
                    self.cx(products[lowest_different_bit - 1],
                            products[lowest_different_bit])

                # update the negation of controls, but no need to do that below lowest_different_bit
                for j in range(lowest_different_bit, self.nb_controls):
                    if ind[j] != prev_ind[j]:
                        self.x(controls[j])

                # rewrite ancillas
                #for j in range(1, self.nb_controls):
                #    self.ccx( controls[j], products[j-1], products[j])

                for j in range(lowest_different_bit + 1, self.nb_controls):
                    self.ccx(controls[j], products[j - 1], products[j])

            # load the data. To optimize the circuit, we actually perform cnots
            # at different levels to account for bits which remain the same
            # between successive indices (this was precomputed above)
            for j in range(self.bit_size):
                for k in do_cnot:
                    if do_cnot[k][j][i] == 1:
                        self.cx(products[-k], output_reg[j])

        # erase ancillas finally
        for j in reversed(range(self.nb_controls - 2)):
            self.ccx(controls[j + 2], ancilla[j], ancilla[j + 1])
        self.ccx(controls[0], controls[1], ancilla[0])

        # revert negation of controls
        for j in range(self.nb_controls):
            self.x(controls[j])

    def test(self):
        for _ in range(20):
            c = [random.randrange(2) for _ in range(self.nb_controls)]
            y = d[tuple(c)]
            input_bits = c + [0] * (self.bit_size + self.ancilla_nbr)
            output_bits = simulate(self, input_bits)
            expected_output = c + int_to_bits(
                y, width=self.bit_size) + [0] * (self.ancilla_nbr)
            assert output_bits == expected_output
        print(self.name, "test passed")


#================================================


def list_to_dict(l, p):
    """(Used in ControlledModularProduct)."""
    nb_controls = len(l)
    d = dict()
    for i in range(1 << nb_controls):
        c = int_to_bits(i, width=nb_controls)
        prod = 1
        for j in range(nb_controls):
            if c[j]:
                prod = (prod * l[j]) % p
        d[tuple(c)] = prod
    return d


class ControlledModularProduct(QuantumCircuit):
    """Performs a controlled modular product of integers (the input bits determine
    if the precomputed integers are present or not).
    
    It is optimized for a product of 21 numbers modulo a prime of 21 bits. We
    divide the product into 3 groups and use table lookups.
    """

    def __init__(self, l, p, controlled=False):
        """Creates the circuit.
        
        Args:
        - l: the list of numbers to be multiplied
        - p: the prime
        - controlled: if True, the circuit will be controlled on the first bit.
        
        """
        super().__init__(name="controlled_mod_product")
        self.p = p
        self.l = l
        log2p = ceil(log(p, 2))
        self.log2p = log2p
        self.nb_controls = len(l)
        self.controlled = controlled

        _tmp = self.nb_controls // 3  # group size for lookups
        #assert self.nb_controls <= 21
        if self.nb_controls <= 15:
            raise ValueError("Unsupported")
        l1, l2, l3 = l[:_tmp], l[_tmp:(2 * _tmp)], l[(2 * _tmp):]
        d1, d2 = list_to_dict(l1, p), list_to_dict(l2, p)
        d3 = list_to_dict(l3, p)
        control = QuantumRegister(1)
        controls = QuantumRegister(self.nb_controls)
        controls1 = controls[:_tmp]
        controls2 = controls[_tmp:(2*_tmp)]
        controls3 = controls[(2*_tmp):]

        load1 = TableLookup(d1, log2p)
        load2 = TableLookup(d2, log2p)
        load3 = TableLookup(d3, log2p)

        # by definition rc3_reg contains enough space for all the ancillas
        # of lookup circuits

        mult1 = Multiplier(log2p, log2p)  # x y xy
        #     Remainder appears in the input register. Quotient appears in the second
        # register of size (bit_size - p_bit_size + 1)
        divider = EuclideanDivider(p, 2 * log2p)

        c1_reg = QuantumRegister(log2p)
        c2_reg = QuantumRegister(log2p)
        c1c2_reg = QuantumRegister(2 * log2p)
        rc3_reg = QuantumRegister(2 * log2p)
        quotient_pad_1 = QuantumRegister(2)
        quotient_pad_2 = QuantumRegister(2)

        out_reg = QuantumRegister(log2p)

        ancillas = QuantumRegister(
            max([
                _c.ancilla_nbr for _c in [load1, load2, load3, mult1, divider]
            ]))

        self.ancilla_nbr = (len(ancillas) + len(c1_reg) + len(c2_reg) +
                            len(c1c2_reg) + len(rc3_reg) +
                            len(quotient_pad_1) + len(quotient_pad_2))

        if self.controlled:
            self.add_register(control)
        self.add_register(controls, out_reg, c1_reg, c2_reg, c1c2_reg,
                          quotient_pad_1, rc3_reg, quotient_pad_2, ancillas)
        #=============================
        # step 1: load c1, c2, c3 with lookups
        self.append(load1,
                    controls1[:] + c1_reg[:] + rc3_reg[:load1.ancilla_nbr])
        self.append(load2,
                    controls2[:] + c2_reg[:] + rc3_reg[-load2.ancilla_nbr:])

        # first product + reduction
        self.append(
            mult1,
            c1_reg[:] + c2_reg[:] + c1c2_reg[:] + ancillas[:mult1.ancilla_nbr])
        self.append(
            divider,
            c1c2_reg[:] + quotient_pad_1[:] + ancillas[:divider.ancilla_nbr])

        # erase c2, load c3 (in parallel)
        self.append(load2,
                    controls2[:] + c2_reg[:] + rc3_reg[:load2.ancilla_nbr])
        self.append(load3,
                    controls3[:] + c2_reg[:] + rc3_reg[:load3.ancilla_nbr:])

        # second product + reduction
        self.append(
            mult1, c2_reg[:] + c1c2_reg[:log2p] + rc3_reg[:] +
            ancillas[:mult1.ancilla_nbr])
        self.append(
            divider,
            rc3_reg[:] + quotient_pad_2[:] + ancillas[:divider.ancilla_nbr])

        # copy remainder to output, but only if control contains 1
        for i in range(self.log2p):
            if self.controlled:
                self.ccx(control, rc3_reg[i], out_reg[i])
            else:
                self.cx(rc3_reg[i], out_reg[i])

        self.append(
            divider.inverse(),
            rc3_reg[:] + quotient_pad_2[:] + ancillas[:divider.ancilla_nbr])
        self.append(
            mult1.inverse(), c2_reg[:] + c1c2_reg[:log2p] + rc3_reg[:] +
            ancillas[:mult1.ancilla_nbr])

        # then c1c2 is 0
        self.append(load3,
                    controls3[:] + c2_reg[:] + rc3_reg[:load3.ancilla_nbr])
        self.append(load2,
                    controls2[:] + c2_reg[:] + rc3_reg[:load2.ancilla_nbr])

        self.append(
            divider.inverse(),
            c1c2_reg[:] + quotient_pad_1[:] + ancillas[:divider.ancilla_nbr])
        self.append(
            mult1.inverse(),
            c1_reg[:] + c2_reg[:] + c1c2_reg[:] + ancillas[:mult1.ancilla_nbr])

        self.append(load1,
                    controls1[:] + c1_reg[:] + rc3_reg[:load1.ancilla_nbr])
        self.append(load2,
                    controls2[:] + c2_reg[:] + rc3_reg[-load2.ancilla_nbr:])

    def test(self):
        qc = self  #full_decompose(self)
        for _ in range(20):
            c = [random.randrange(2) for _ in range(self.nb_controls)]
            prod = 1
            for i in range(self.nb_controls):
                if c[i] == 1:
                    prod = (prod * self.l[i]) % self.p
            input_bits = c + [0] * (self.log2p + self.ancilla_nbr)
            output_bits = simulate(qc, input_bits)
            expected_output = c + int_to_bits(
                prod, width=self.log2p) + [0] * (self.ancilla_nbr)
            assert output_bits == expected_output
        print(self.name, "test passed")


if __name__ == "__main__":

    bit_size = 20
    w = 7
    d = {
        tuple(int_to_bits(i, width=w)): random.randrange(1 << bit_size)
        for i in range(1 << w)
    }
    qc = TableLookup(d, bit_size=bit_size)
    qc.test()
    qc = full_decompose(qc, do_not_decompose=[])
    print(gate_counts(qc))
    print(qc.depth())

    #qc = EuclideanDivider(17, 10)
    #qc.test()

    #qc = Incrementor(4)
    #qc.test()
