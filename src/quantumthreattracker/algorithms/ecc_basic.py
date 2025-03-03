"""Class for a basic implementation of ECC.

Logical resource estimates taken from [1], Table VI.

[1] https://doi.org/10.48550/arXiv.2409.04643
"""

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import QuantumAlgorithm


class ECCBasic(QuantumAlgorithm):
    """Class for a basic implementation of ECC."""

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

        qubit_count = 9 * key_size
        toffoli_count = 43 * key_size**3

        alg_sum = AlgorithmSummary(
            n_algo_qubits=qubit_count, n_logical_gates=GateCounts(toffoli=toffoli_count)
        )

        return alg_sum
