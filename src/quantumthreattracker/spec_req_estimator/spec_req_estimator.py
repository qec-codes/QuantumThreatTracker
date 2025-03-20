"""Class for estimmating quantum computer specification requirements."""

from qsharp.estimator import EstimatorError, EstimatorParams

from quantumthreattracker.algorithms import (
    CryptParams,
    GidneyEkeraBasic,
    QuantumAlgorithm,
)


def update_estimator_params(estimator_params: dict, gate_error_rate: float) -> dict:
    """Update the estimator parameters with a new gate error rate.

    Parameters
    ----------
    estimator_params : dict
        Estimator parameters.
    gate_error_rate : float
        New gate error rate.

    Returns
    -------
    dict
        Estimator parameters with updated gate error rate.
    """
    if "qubitParams" not in estimator_params:
        estimator_params["qubitParams"] = {}

    if "instructionSet" not in estimator_params["qubitParams"]:
        estimator_params["qubitParams"]["instructionSet"] = "GateBased"
    if "oneQubitMeasurementTime" not in estimator_params["qubitParams"]:
        estimator_params["qubitParams"]["oneQubitMeasurementTime"] = "100 ns"
    if "oneQubitGateTime" not in estimator_params["qubitParams"]:
        estimator_params["qubitParams"]["oneQubitGateTime"] = "50 ns"
    if "twoQubitGateTime" not in estimator_params["qubitParams"]:
        estimator_params["qubitParams"]["twoQubitGateTime"] = "50 ns"
    if "tGateTime" not in estimator_params["qubitParams"]:
        estimator_params["qubitParams"]["tGateTime"] = "50 ns"
    estimator_params["qubitParams"]["oneQubitMeasurementErrorRate"] = gate_error_rate
    estimator_params["qubitParams"]["oneQubitGateErrorRate"] = gate_error_rate
    estimator_params["qubitParams"]["twoQubitGateErrorRate"] = gate_error_rate
    estimator_params["qubitParams"]["tGateErrorRate"] = gate_error_rate
    return estimator_params


class SpecReqEstimator:
    """Class for estimmating quantum computer specification requirements."""

    def __init__(self, crypt_params: CryptParams):
        self._crypt_params = crypt_params

    def estimate_specs(
        self,
        estimator_params: EstimatorParams | dict,
        quantum_algorithm: type[QuantumAlgorithm] | None = None,
    ) -> float:
        """Estimate the gate error rate needed for a given number of qubits.

        Parameters
        ----------
        estimator_params : EstimatorParams | dict
            Parameters for Azure's quantum resource estimator. The 'max_physical_qubits'
            parameter should be defined.
        quantum_algorithm : QuantumAlgorithm, optional
            Quantum algorithm to use for the analysis, by default None.

        Returns
        -------
        float
            Required gate error rate.

        Raises
        ------
        TypeError
            If the estimator parameters are of the wrong type.
        RuntimeError
            If the function fails to find a sufficiently low gate error rate.
        """
        if isinstance(estimator_params, EstimatorParams):
            estimator_params = estimator_params.as_dict()
        elif not isinstance(estimator_params, dict):
            raise TypeError(
                f"{type(estimator_params)} is the wrong type for estimator parameters. "
                + "It must be given as either an EstimatorParams instance or a dictionary."
            )

        # TODO: replace this with the "algorithm picker" routine, once it exists
        if quantum_algorithm is None:
            quantum_algorithm = GidneyEkeraBasic(crypt_params=self._crypt_params)
        else:
            quantum_algorithm = quantum_algorithm(crypt_params=self._crypt_params)

        # Finding a lower bound on the required gate error rate.
        min_gate_error_rate = 1e-50
        gate_error_rate = 0.001
        lower_bound_found = False
        while lower_bound_found is False:
            if gate_error_rate < min_gate_error_rate:
                raise RuntimeError("Could not find a sufficiently low gate error rate.")
            try:
                estimator_result = quantum_algorithm.estimate_resources_azure(
                    estimator_params
                )
                lower_bound_found = True
            except EstimatorError:
                gate_error_rate /= 10
                estimator_params = update_estimator_params(
                    estimator_params, gate_error_rate
                )

        # Performing a binary search to refine the exact gate error rate required.
        binary_search_num_steps = 10
        gate_error_rate_min = gate_error_rate
        gate_error_rate_max = 10 * gate_error_rate
        for _ in range(binary_search_num_steps):
            gate_error_rate = gate_error_rate_min + 0.5 * (
                gate_error_rate_max - gate_error_rate_min
            )
            estimator_params = update_estimator_params(
                estimator_params, gate_error_rate
            )
            try:
                estimator_result = quantum_algorithm.estimate_resources_azure(
                    estimator_params
                )
                gate_error_rate_min = gate_error_rate
            except EstimatorError:
                gate_error_rate_max = gate_error_rate
        gate_error_rate = gate_error_rate_min

        return {
            "gateErrorRate": gate_error_rate,
            "estimatorResult": estimator_result,
        }
