"""Resource estimator for controlled multi product RSA algorithm."""
from dataclasses import dataclass
from typing import Optional

from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)

from .chevignard_utils.estimations import full_circuit_costs


@dataclass
class ChevignardParams(AlgParams):
    """(No tunable parameters for Chevignard; placeholder.)."""

    pass


class Chevignard(QuantumAlgorithm):
    """Qualtran wrapper for Chevignard et. al's ctrl multi product RSA algorithm."""

    def __init__(
        self, crypt_params: CryptParams, alg_params: Optional[ChevignardParams] = None
    ):
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Estimate the resource requirements for controlled multi product.

        Raises
        ------
            NameError: Chevignard only supports protocol "RSA".

        Args
        ----
            alg_params (AlgParams): Algorithm parameters. Not used in this case.

        Returns
        -------
            AlgorithmSummary: Resource requirements for the algorithm.

        Raises
        ------
        NameError
            If the protocol is not "RSA".
        """
        if self._crypt_params.protocol != "RSA":
            raise NameError(
                'Chevignard only supports protocol "RSA", got '
                f'"{self._crypt_params.protocol}"'
            )

        # 1) compute all RNS parameters for this key size
        n = self._crypt_params.key_size
        return full_circuit_costs(n)

    @staticmethod
    def generate_search_space() -> list[ChevignardParams]:
        """
        Generate the search space for the Chevignard algorithm.

        Returns
        -------
            list[ChevignardParams]: A list of ChevignardParams instances defining
                the parameter configurations to explore.
        """
        return [ChevignardParams()]
