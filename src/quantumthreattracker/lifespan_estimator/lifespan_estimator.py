"""Class for estimating the remaining lifespan of cryptographic protocols."""

import copy
import json
from pathlib import Path

from qsharp.estimator import EstimatorError

from quantumthreattracker.algorithms.baseline_shor import (
    BaselineShor,
    BaselineShorParams,
)
from quantumthreattracker.lifespan_estimator.hardware_roadmap import HardwareRoadmap


class LifespanEstimator:
    """Class for estimating the remaining lifespan of cryptographic protocols."""

    def __init__(self, hardware_roadmap: HardwareRoadmap):
        self._hardware_roadmap = hardware_roadmap
        self._threat_report = None

    def estimate_threats(self, protocol: str, key_size: int) -> dict:
        """Estimate the possible threats against a given cryptographic protocol.

        Parameters
        ----------
        protocol : str
            Cryptographic protocol.
        key_size : int
            Cryptographic key size.

        Returns
        -------
        dict
            Threats against the given protocol.

        Raises
        ------
        SyntaxError
            If the given detail level is not within the required bounds.
        """
        # TODO:  Once we have more algorithms implemented, there should be an additional
        # module here to choose which quantum algorithm to use based on the
        # cryptographic protocol, and perhaps user input.
        algorithm_params = BaselineShorParams(protocol=protocol, key_size=key_size)
        algorithm = BaselineShor(algorithm_params=algorithm_params)

        threats = []

        for milestone in self._hardware_roadmap.as_list():
            for quantum_computer in milestone["hardwareList"]:
                timestamp = milestone["timestamp"]
                try:
                    estimator_result = algorithm.estimate_resources_azure(
                        quantum_computer["estimatorParams"]
                    )
                    threats.append(
                        {
                            "timestamp": timestamp,
                            "estimatorResult": estimator_result,
                        }
                    )
                except EstimatorError:
                    pass

        return {
            "protocol": str(protocol) + "-" + str(key_size),
            "threats": threats,
        }

    def generate_report(self, protocols: list[dict]):
        """Predict the threats against several cryptographic protocols.

        Parameters
        ----------
        protocols : list[dict]
            List of cryptographic protocols.
        """
        threat_report = []
        for protocol_and_key_size in protocols:
            threat_report.append(
                self.estimate_threats(
                    protocol_and_key_size["algorithm"],
                    protocol_and_key_size["keySize"],
                )
            )

        self._threat_report = threat_report

    def get_report(self, detail_level: int = 3) -> list:
        """Get the threat report.

        Parameters
        ----------
        detail_level : int, optional
            Level of detail in the output, by default 3.

        Returns
        -------
        list
            Threat report.

        Raises
        ------
        AttributeError
            If the report has not yet been generated.
        SyntaxError
            If the detail level is out of bounds.
        """
        if self._threat_report is None:
            raise AttributeError(
                "The threat report has not been generated and thus cannot be saved."
            )

        report_output = copy.deepcopy(self._threat_report)

        for report_entry in report_output:
            if detail_level == 0:
                report_entry["threats"] = [
                    {"timestamp": threat["timestamp"]}
                    for threat in report_entry["threats"]
                ]
            elif detail_level == 1:
                report_entry["threats"] = [
                    {
                        "timestamp": threat["timestamp"],
                        "runtime": threat["estimatorResult"]["physicalCounts"][
                            "runtime"
                        ],
                    }
                    for threat in report_entry["threats"]
                ]
            elif detail_level == 2:
                report_entry["threats"] = [
                    {
                        "timestamp": threat["timestamp"],
                        "physicalCounts": threat["estimatorResult"]["physicalCounts"],
                    }
                    for threat in report_entry["threats"]
                ]
            elif detail_level != 3:
                raise SyntaxError(
                    f"Detail level ({detail_level}) must be an integer between 0 and 3 (inclusive)."
                )
        return report_output

    def save_report(self, file_name: str, file_path: str = None, detail_level: int = 3):
        """Save the hardware roadmap as a JSON file.

        Parameters
        ----------
        file_name : str
            File name.
        file_path : str, optional
            File path. If unspecified, the file will be saved to the current working
            directory.
        detail_level : int, optional
            Level of detail in the output, by default 3.
        """
        report_output = self.get_report(detail_level=detail_level)
        if file_path is None:
            file_path = str(Path.cwd())
        with Path.open(file_path + "/" + file_name + ".json", "w") as fp:
            json.dump(report_output, fp, indent=4)
