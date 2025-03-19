"""Class for a basic implementation of Gidney-Ekera."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)


@dataclass
class GidneyEkeraBasicParams(AlgParams):
    """Parameters for the basic implementation of Gidney-Ekera.

    This implementation doesn't have configurable parameters,
    but this class is provided for consistency with the interface.
    """

    pass


class GidneyEkeraBasic(QuantumAlgorithm):
    """Class for a basic implementation of Gidney-Ekera."""

    def __init__(self, crypt_params: CryptParams, alg_params: Optional[GidneyEkeraBasicParams] = None):
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

    def get_algorithm_summary(self, alg_params: Optional[AlgParams] = None) -> AlgorithmSummary:
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
        """
        # params check is unnecessary for GidneyEkeraBasic as it doesn't use any
        # But we'll include it for consistency with the interface
        # effective_alg_params = alg_params or self._alg_params

        if self._crypt_params.protocol != "RSA":
            raise NameError(
                'The protocol for this class must be "RSA". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        key_size = self._crypt_params.key_size

        qubit_count = int(np.ceil(3 * key_size + 0.002 * key_size * np.log(key_size)))
        toffoli_count = int(
            np.ceil(0.3 * key_size**3 + 0.0005 * key_size**3 * np.log(key_size))
        )
        measurement_depth = int(
            np.ceil(500 * key_size**2 + key_size**2 * np.log(key_size))
        )

        alg_sum = AlgorithmSummary(
            n_algo_qubits=qubit_count,
            n_logical_gates=GateCounts(
                toffoli=toffoli_count, measurement=measurement_depth
            ),
        )
        return alg_sum

    def generate_search_space(self) -> list[GidneyEkeraBasicParams]:
        """Generate a search space for algorithm parameters.

        Since GidneyEkeraBasic doesn't have configurable parameters, this returns
        a list with a single set of default parameters.

        Returns
        -------
        list[GidneyEkeraBasicParams]
            Single-element list containing default parameters.
        """
        return [GidneyEkeraBasicParams()]
