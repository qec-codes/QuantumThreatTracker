"""Class for a parameterised implementation of Gidney (2025)'s implementation.

[1] https://arxiv.org/abs/2505.15917
"""

from dataclasses import dataclass
from typing import Optional

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)


@dataclass
class GidneyBasicParams(AlgParams):
    """(No tunable parameters for this basic implementation; placeholder.)."""

    pass


class GidneyBasic(QuantumAlgorithm):
    """Class for a basic implementation of Gidney."""

    def __init__(
        self,
        crypt_params: CryptParams,
        alg_params: Optional[GidneyBasicParams] = None,
    ):
        """Initialize the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[GidneyEkeraBasicParams], optional
            Algorithmic parameters. For the basic implementation, these have no effect
            but are included for API consistency.
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters (not used in this basic implementation)

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NameError
            If the protocol is not "RSA".
        ValueError
            If the key_size is not supported.
        """
        if self._crypt_params.protocol != "RSA":
            raise NameError(
                'The protocol for this class must be "RSA". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        key_size = self._crypt_params.key_size

        # Table mapping key_size (n) to Toffolis and Qubits
        resource_table = {
            1024: {"qubit_count": 742, "toffoli_count": 1.1e9},
            1536: {"qubit_count": 1074, "toffoli_count": 3.1e9},
            2048: {"qubit_count": 1399, "toffoli_count": 6.5e9},
            3072: {"qubit_count": 2043, "toffoli_count": 1.9e10},
            4096: {"qubit_count": 2692, "toffoli_count": 4.0e10},
            6144: {"qubit_count": 3978, "toffoli_count": 1.2e11},
            8192: {"qubit_count": 5261, "toffoli_count": 2.7e11},
        }

        sorted_keys = sorted(resource_table.keys())
        key_size_rounded_up = None
        for k in sorted_keys:
            if key_size <= k:
                key_size_rounded_up = k
                break

        if key_size_rounded_up is None:
            raise ValueError(
                f"Unsupported key_size: {key_size} exceeds maximum supported ({sorted_keys[-1]})"
            )

        key_size = key_size_rounded_up

        qubits = int(resource_table[key_size]["qubit_count"])
        toffolis = int(resource_table[key_size]["toffoli_count"])

        return AlgorithmSummary(
            n_algo_qubits=int(qubits),
            n_logical_gates=GateCounts(toffoli=toffolis),
        )

    @staticmethod
    def generate_search_space() -> list[GidneyBasicParams]:
        """Generate a search space for algorithm parameters.

        Since GidneyBasic doesn't have configurable parameters, this returns
        a list with a single set of default parameters.

        Returns
        -------
        list[GidneyBasicParams]
            Single-element list containing default parameters.
        """
        return [GidneyBasicParams()]
