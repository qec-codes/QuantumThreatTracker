"""Class for a parameterised implementation of Gidney-Ekera.

https://doi.org/10.22331/q-2021-04-15-433
"""

from dataclasses import dataclass

from qualtran import QUInt
from qualtran.bloqs.arithmetic import Add
from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    CryptParams,
    QuantumAlgorithm,
)


@dataclass
class GidneyEkeraParams:
    num_exp_qubits: int
    num_pad_qubits: int
    window_size_exp: int
    window_size_mul: int


class GidneyEkeraAdvanced(QuantumAlgorithm):
    def __init__(self, crypt_params: CryptParams, alg_params: GidneyEkeraParams):
        super().__init__(crypt_params)
        self._alg_params = alg_params

    def get_algorithm_summary(self):
        key_size = self._crypt_params.key_size
        num_exp_qubits = self._alg_params.num_exp_qubits
        num_pad_qubits = self._alg_params.num_pad_qubits
        window_size_exp = self._alg_params.window_size_exp
        window_size_mul = self._alg_params.window_size_mul

        num_lookup_additions = int(
            2 * key_size * num_exp_qubits / (window_size_exp * window_size_mul)
        )

        adder = Add(a_dtype=QUInt(key_size), b_dtype=QUInt(key_size))
        lookup_cost = GateCounts(toffoli=int(2 ** (window_size_exp + window_size_mul)))

        logical_qubit_count = 3 * key_size

        total_gate_count = num_lookup_additions * (
            get_cost_value(adder, QECGatesCost()) + lookup_cost
        )

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )
