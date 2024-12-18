"""Base class for quantum algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass
class QuantumAlgorithmParameters(ABC):
    """Base dataclass for quantum algorithm parameters.

    Parameters
    ----------
    protocol: str
        Cryptographic protocol. Can be:
            - 'RSA' (Rivest-Shamir-Adleman; factoring)
            - 'DH' (Diffie-Helllman; discrete log)
            - 'ECDH' (Elliptic Curve Diffie-Hellman; discrete log over elliptic curves)
    key_size: int
        Cryptographic key size.
    """

    protocol: str
    key_size: int


class QuantumAlgorithm(ABC):
    """Base class for quantum algorithms."""

    @abstractmethod
    def success_probability(self) -> float:
        """Calculate the algorithmic success probability.

        Returns
        -------
        float
            Algorithmic success probability.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def _target_unitary(self) -> QuantumCircuit:
        """Construct the target unitary to perform period finding on.

        Returns
        -------
        QuantumCircuit
            Target unitary.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def create_qiskit_circuits(self) -> list[tuple[QuantumCircuit, int]]:
        """Explicitly construct all quantum circuits used in the algorithm.

        Returns
        -------
        list[tuple[QuantumCircuit, int]]
            List of quantum circuits, together with the number of repetitions required
            for each circuit.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError
