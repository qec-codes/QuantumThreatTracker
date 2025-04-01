"""Class for the implementation of Ekerå-Håstad alg for dlog in safe prime groups."""

from dataclasses import dataclass
from typing import Optional

from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)
from quantumthreattracker.algorithms.utils import (
    fips_strength_level_rounded,
    generalize_and_decomp,
)


@dataclass
class DLogSafePrimeEHParams(AlgParams):
    """Parameters for discrete logarithm implementation using Ekerå-Håstad algorithm.

    In discrete logarithm problems with safe-prime groups, the order is n-1
    where n is the modulus bit length. With short exponent: nd = 2z.

    Note: This implementation focuses only on short exponents in safe-prime groups
    using the Ekerå-Håstad approach.
    """

    window_size_exp: int
    window_size_mul: int


class DLogSafePrimeEH(QuantumAlgorithm):
    """Class for the implementation of Ekerå-Håstad algorithm.

    for dlog problems in safe prime groups.

    Method based on Ekerå-Håstad's improvements to Shor's algorithm.
    """

    def __init__(
        self, crypt_params: CryptParams, alg_params: Optional[DLogSafePrimeEHParams] = None
    ):
        """Initialize the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[DLogSafePrimeEHParams], optional
            Algorithmic parameters for discrete logarithm problems.
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NameError
            If the protocol is not "DH-SP".
        TypeError
            If alg_params is not of type DLogSafePrimeEHParams.
        """
        if self._crypt_params.protocol != "DH-SP":
            raise NameError(
                'The protocol for this class must be "DH-SP". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        # Use provided alg_params or instance alg_params
        effective_alg_params = alg_params or self._alg_params

        # Type checking
        if (
            effective_alg_params is not None
            and not isinstance(effective_alg_params, DLogSafePrimeEHParams)
        ):
            raise TypeError(
                f"Expected DLogSafePrimeEHParams, got {type(effective_alg_params).__name__}"
            )

        key_size = self._crypt_params.key_size
        window_size_exp = effective_alg_params.window_size_exp
        window_size_mul = effective_alg_params.window_size_mul

        # Main difference is in how num_exp_qubits is calculated
        # For EH algorithm with safe prime groups and short exponents
        z = fips_strength_level_rounded(key_size)
        num_exp_qubits = 6 * z

        # TODO: remove the custom overriding of the `Add` bloqs once the Qualtran
        # implementation is fixed.
        adder = Add(a_dtype=QUInt(key_size), b_dtype=QUInt(key_size))
        adder_bloq_counts = adder.call_graph(
            max_depth=10, generalizer=generalize_and_decomp
        )[1]
        adder_cost = GateCounts(
            toffoli=adder_bloq_counts[Toffoli()],
            clifford=adder_bloq_counts[CNOT()],
        )

        lookup_cost = GateCounts(
            toffoli=int(2 ** (window_size_exp + window_size_mul))
        )

        num_lookup_additions = int(
            2 * key_size * num_exp_qubits / (window_size_exp * window_size_mul)
        )
        total_gate_count = num_lookup_additions * (lookup_cost + adder_cost)

        logical_qubit_count = 3 * key_size

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )

    @staticmethod
    def generate_search_space() -> list[DLogSafePrimeEHParams]:
        """Generate a search space for algorithm parameters.

        Returns
        -------
        list[DLogSafePrimeEHParams]
            List of parameters for different window size combinations.
        """
        search_space = []

        for window_size_exp in [2, 3, 4, 5, 6, 7]:
            for window_size_mul in [2, 3, 4, 5, 6, 7]:
                params = DLogSafePrimeEHParams(
                    window_size_exp=window_size_exp,
                    window_size_mul=window_size_mul,
                )
                search_space.append(params)
        return search_space
