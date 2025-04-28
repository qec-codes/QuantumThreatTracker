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
Contains some useful functions to deal with classical circuits.

Implemented using Qiskit's QuantumCircuit: simulating the circuits,
decomposing them into elementary classical gates and counting the gates.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class GateCounts(dict):
    """A dictionary for gate counts of a reversible circuit."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__()
        self['x'] = 0
        self['cx'] = 0
        self['ccx'] = 0
        self['swap'] = 0
        for (k, v) in kwargs.items():
            if k in self:
                self[k] = v

    def __mul__(self, other: int) -> "GateCounts":
        """
        Multiply all gate counts by an integer.

        Returns
        -------
            GateCounts: A new GateCounts object with counts multiplied by the integer.
        """
        result = GateCounts()
        for key in self:
            result[key] = self[key] * other
        return result

    def __add__(self, other: "GateCounts") -> "GateCounts":
        """
        Add gate counts from another GateCounts object.

        Returns
        -------
            GateCounts: A new GateCounts object with summed gate counts.
        """
        result = GateCounts()
        for key in self:
            result[key] = self[key] + other[key]
        return result

    def __le__(self, other: "GateCounts") -> bool:
        """Compare gate count is LEQ another GateCounts object.

        Returns
        -------
            bool: True if all gate counts are less than or equal, False otherwise.
        """
        for key in self:
            if other[key] < self[key]:
                return False
        return True

    def __ge__(self, other: "GateCounts") -> bool:
        """Compare gate count is GEQ another GateCounts object.

        Returns
        -------
            bool: True if all gate counts are greater than or equal, False otherwise.
        """
        for key in self:
            if other[key] > self[key]:
                return False
        return True


def int_to_bits(i: int, width: int = 4, rev: bool = True) -> list:
    """Convert integer to list of bits.

    Returns
    -------
        list: List of bits (0 or 1) representing the integer.
    """
    if rev:
        return list(
            reversed([int(c) for c in '{:0{width}b}'.format(i, width=width)]))
    else:
        return list([int(c) for c in '{:0{width}b}'.format(i, width=width)])


def bits_to_int(s: list, rev: bool = True) -> int:
    """Convert list of bits to integer.

    Returns
    -------
        int: Integer value represented by the list of bits.
    """
    if rev:
        return int(''.join(list(reversed([str(b) for b in s]))), 2)
    else:
        return int(''.join(list([str(b) for b in s])), 2)


class DummyGate(Gate):
    """A dummy gate that does nothing. Used for debugging."""

    def __init__(self, num_qubits: int) -> None:
        super().__init__(name="dummy", num_qubits=num_qubits, params=[])

    def apply(self, input_bits):  # noqa: ANN001, ANN201, D102
        raise NotImplementedError()


class PrintGate(DummyGate):
    """A gate that prints the bits of its register, during the simulation.

    Very useful for debugging.
    """

    def __init__(self, num_qubits: int, myname: str = "") -> None:
        self.myname = myname
        super().__init__(num_qubits)

    def apply(self, input_bits):  # noqa: ANN001, ANN201, D102
        print("====", self.myname)
        print(input_bits)
        return input_bits[:]


class PrintIntGate(DummyGate):
    """A gate that prints the int value contained in its register.

    Very useful for debugging.
    """

    def __init__(self, num_qubits: int, myname: str = ""):
        self.myname = myname
        super().__init__(num_qubits)

    def apply(self, input_bits: list) -> list:  # noqa: D102
        print("====", self.myname)
        print(bits_to_int(input_bits))
        return input_bits[:]


def simulate(qc: QuantumCircuit, inp_bits: list) -> list:
    """Simulate a (classical) quantum circuit on given input bits.

    Args
    ----
        qc (QuantumCircuit): The circuit to simulate.
        inp_bits (list): The input bits.

    Returns
    -------
        list: The result of the simulation as a list of bits.

    Raises
    ------
    ValueError
        If the # of input bits does not match the # qubits in circuit.
    """
    if len(qc.qubits) != len(inp_bits):
        raise ValueError("Expected %i input bits, got %i" %
                         (len(qc.qubits), len(inp_bits)))
    res = inp_bits[:]  # copy input
    # apply all gates of the circuit
    for instr in qc.data:
        op = instr.operation
        qubits = instr.qubits
        qubit_indices = [qc.find_bit(q).index for q in qubits]
        op_input = [res[t] for t in qubit_indices]
        if op.name == "dummy":
            new_q = op.apply(op_input)
            for i in range(len(qubit_indices)):
                res[qubit_indices[i]] = new_q[i]
        elif op.name == "swap":
            a, b = tuple(qubit_indices)
            tmp = res[a]
            res[a] = res[b]
            res[b] = tmp
        elif op.name == "x":
            a = qubit_indices[0]
            res[a] ^= 1
        elif op.name == "cx":
            a, b = tuple(qubit_indices)
            res[b] ^= res[a]
        elif op.name == "ccx":
            a, b, c = tuple(qubit_indices)
            res[c] ^= (res[a] & res[b])
        elif isinstance(op._definition, QuantumCircuit):
            # access _definition, which is another circuit
            new_q = simulate(op._definition, op_input)
            for i in range(len(qubit_indices)):
                res[qubit_indices[i]] = new_q[i]
        else:
            raise ValueError("Unsupported gates in circuit")
    return res


def gate_counts(qc: QuantumCircuit) -> GateCounts:
    """Count the gates of a (classical) QuantumCircuit.

    Args
    ----
        qc (QuantumCircuit): The circuit to count.

    Returns
    -------
        GateCounts: A dictionary with the counts of each gate.

    Raises
    ------
    ValueError
            If the circuit contains unsupported gates.
    """
    res = GateCounts()
    for instr in qc.data:
        op = instr.operation
        if op.name in {"dummy", "swap", "x", "cx", "ccx"}:
            if op.name not in res:
                res[op.name] = 0
            res[op.name] += 1
        elif isinstance(op._definition, QuantumCircuit):
            # access _definition, which is another circuit
            tmp = gate_counts(op._definition)
            res += tmp


#            for k in tmp:
#                if k not in res:
#                    res[k] = 0
#                res[k] += tmp[k]
        else:
            raise ValueError("Unsupported gates in circuit")
    return res


def full_decompose(qc: QuantumCircuit, do_not_decompose: list | None = None) -> QuantumCircuit:
    """Decompose a QuantumCircuit into elementary (classical) gates.

    Returns
    -------
        QuantumCircuit: A new circuit with the decomposed gates.

    Raises
    ------
    ValueError
        If the circuit contains unsupported gates.
    """
    if do_not_decompose is None:
        do_not_decompose = []

    def _do_not_decompose(name: str) -> bool:
        for s in do_not_decompose:
            if name.startswith(s):
                return True
        return False

    res = QuantumCircuit(len(qc.qubits))

    for instr in qc.data:
        op = instr.operation
        qubits = instr.qubits
        # index in qc of all the qubits of instr (in order)
        qubit_indices = [qc.find_bit(q).index for q in qubits]

        if op.name in {"swap", "x", "cx", "ccx"}:
            res.append(op, [res.qubits[i] for i in qubit_indices])

        elif _do_not_decompose(op.name):
            res.append(op._definition, [res.qubits[i] for i in qubit_indices])

        elif isinstance(op._definition, QuantumCircuit):
            # access _definition, which is another circuit
            new_qc = full_decompose(op._definition,
                                    do_not_decompose=do_not_decompose)
            for new_instr in new_qc.data:
                new_op = new_instr.operation
                new_qubits = new_instr.qubits
                # index in new_qc of all the qubits of new_instr (in order)
                # qubit
                new_qubit_indices = [
                    new_qc.find_bit(q).index for q in new_qubits
                ]

                if new_op.name in {"swap", "x", "cx", "ccx"}:
                    res.append(new_op, [
                        res.qubits[qubit_indices[i]] for i in new_qubit_indices
                    ])
                elif _do_not_decompose(new_op.name):
                    res.append(new_op._definition, [
                        res.qubits[qubit_indices[i]] for i in new_qubit_indices
                    ])
                else:
                    raise ValueError("Unsupported gates in circuit")
        else:
            print(op)
            print(op._definition)
            raise ValueError("Unsupported gates in circuit")
    return res
