"""Class for a basic implementation of Gidney-Ekera.

Creates physical resource estimates from the logical resource counts given in the
abstract.

https://doi.org/10.22331/q-2021-04-15-433
"""

import numpy as np
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import QuantumAlgorithm


class GidneyEkeraBasic(QuantumAlgorithm):
    """Class for a basic implementation of Gidney-Ekera."""

    def success_probability(self):
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
        return super().success_probability()

    def get_algorithm_summary(self):
        """Compute logical resource estimates for the circuit.

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
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
