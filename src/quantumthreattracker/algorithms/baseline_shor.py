"""Baseline Shor's algorithm."""

from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

from quantumthreattracker.algorithms.quantum_algorithm import (
    QuantumAlgorithm,
    QuantumAlgorithmParameters,
)


@dataclass
class BaselineShorParameters(QuantumAlgorithmParameters):
    """Dataclass for the parameters for baseline Shor's algorithm."""

    pass


class BaselineShor(QuantumAlgorithm):
    """Textbook implementation of Shor's algorithm."""

    def __init__(self, parameters: BaselineShorParameters):
        self._parameters = parameters

    def success_probability(self) -> float:
        """Calculate the algorithmic success probability.

        Returns
        -------
        float
            Algorithmic success probability.
        """
        return super().success_probability()

    def _target_unitary(self) -> QuantumCircuit:
        return super()._target_unitary()

    def create_qiskit_circuits(self) -> list[tuple[QuantumCircuit, int]]:
        """Explicitly construct all quantum circuits used in the algorithm.

        Returns
        -------
        list[tuple[QuantumCircuit, int]]
            List of quantum circuits, together with the number of repetitions required
            for each circuit.
        """
        key_size = self._parameters.key_size

        clock_register_size = 2 * key_size
        arithmetic_register_size = key_size

        clock_register = QuantumRegister(clock_register_size, name="clock")
        arithmetic_register = QuantumRegister(
            arithmetic_register_size, name="arithmetic"
        )

        circuit = QuantumCircuit(clock_register, arithmetic_register)

        circuit.h(clock_register)

        circuit.compose(
            QFT(clock_register_size).inverse(), clock_register, inplace=True
        )

        return [(circuit, 1)]
