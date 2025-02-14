"""Class for estimating the remaining lifespan of cryptographic protocols."""

import json
from pathlib import Path

from qsharp.estimator import EstimatorError, EstimatorParams

from quantumthreattracker.algorithms.baseline_shor import (
    BaselineShor,
    BaselineShorParams,
)


class HardwareRoadmap:
    """Class for representing quantum computing hardware roadmaps."""

    def __init__(self):
        self._hardware_roadmap = []

    def as_list(self) -> list:
        """Get the hardware roadmap as a list.

        Returns
        -------
        list
            Hardware roadmap.
        """
        return self._hardware_roadmap

    def add(
        self,
        timestamp: int,
        estimator_params: EstimatorParams | dict | list,
    ):
        """Add a quantum computer to the hardware roadmap.

        Parameters
        ----------
        timestamp : int
            Unix timestamp.
        estimator_params : EstimatorParams | dict | list
            Parameters characterising the quantum computer.
        """
        if isinstance(estimator_params, list):
            estimator_params_list = estimator_params
        else:
            estimator_params_list = [estimator_params]

        for estimator_params_entry in estimator_params_list:
            if isinstance(estimator_params_entry, EstimatorParams):
                estimator_params_dict = estimator_params_entry.as_dict()
            elif isinstance(estimator_params_entry, dict):
                estimator_params_dict = estimator_params_entry
            else:
                raise TypeError(
                    f"{type(estimator_params_entry)} is the wrong type for estimator parameters. "
                    + "It must be given as either an EstimatorParams instance or a dictionary."
                )

            new_milestone = {
                "timestamp": timestamp,
                "hardwareList": [{"estimatorParams": estimator_params_dict}],
            }

            # This slightly messy chunk of code is for adding the new quantum computer
            # to the existing hardware roadmap as a new entry in the list, or combining
            # it with an existing entry if another entry with the same timestamp already
            # exists.
            inserted = False
            for index in range(len(self._hardware_roadmap)):
                if timestamp < self._hardware_roadmap[index]["timestamp"]:
                    self._hardware_roadmap.insert(index, new_milestone)
                    inserted = True
                    break
                if timestamp == self._hardware_roadmap[index]["timestamp"]:
                    self._hardware_roadmap[index]["hardwareList"].append(
                        new_milestone["hardwareList"][0]
                    )
                    inserted = True
                    break
            if not inserted:
                self._hardware_roadmap.append(new_milestone)

    def remove(self, timestamp: int, qc_index: int = None):
        """Remove an entry from the hardware roadmap.

        Parameters
        ----------
        timestamp : int
            Unix timestamp.
        qc_index : int, optional
            Index for removing a specific quantum computer for a given timestamp. If
            unspecified, all entries with the specified timestamp will be removed.
        """
        for milestone_index in range(len(self._hardware_roadmap)):
            if self._hardware_roadmap[milestone_index]["timestamp"] == timestamp:
                if qc_index is None:
                    self._hardware_roadmap.pop(milestone_index)
                else:
                    self._hardware_roadmap[milestone_index]["hardwareList"].pop(
                        qc_index
                    )
                    if (
                        len(self._hardware_roadmap[milestone_index]["hardwareList"])
                        == 0
                    ):
                        self._hardware_roadmap.pop(milestone_index)
                return

    def save_roadmap(self, file_name: str, file_path: Path = None) -> None:
        """Save the hardware roadmap as a JSON file.

        Parameters
        ----------
        file_name : str
            File name.
        file_path : Path, optional
            File path. If unspecified, the file will be saved to the same directory
            the function is executed from.

        Raises
        ------
        AttributeError
            If the hardware roadmap has not yet been generated.
        """
        if file_path is None:
            file_path = str(Path(__file__).resolve().parent)
        try:
            with Path.open(file_path + "/" + file_name + ".json", "w") as fp:
                json.dump(self._hardware_roadmap, fp, indent=4)
        except AttributeError:
            raise AttributeError(
                "The hardware roadmap has not been generated and thus cannot be saved."
            )


class LifespanEstimator:
    """Class for estimating the remaining lifespan of cryptographic protocols."""

    def __init__(self, hardware_roadmap: HardwareRoadmap):
        self._hardware_roadmap = hardware_roadmap

    def estimate_threats(
        self, protocol: str, key_size: int, detail_level: int = 1
    ) -> dict:
        """Estimate the possible threats against a given cryptographic protocol.

        Parameters
        ----------
        protocol : str
            Cryptographic protocol.
        key_size : int
            Cryptographic key size.
        detail_level : int, optional
            Level of detail in the output. Must be an integer between 0 and 3 inclusive.
            By default 1.

        Returns
        -------
        dict
            Threats against the given protocol.

        Raises
        ------
        SyntaxError
            If the given detail level is not within the required bounds.
        """
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
                    if detail_level == 0:
                        result = {
                            "timestamp": timestamp,
                        }
                    elif detail_level == 1:
                        result = {
                            "timestamp": timestamp,
                            "runtime": estimator_result["physicalCounts"]["runtime"],
                        }
                    elif detail_level == 2:
                        result = {
                            "timestamp": timestamp,
                            "physicalCounts": estimator_result["physicalCounts"],
                        }
                    elif detail_level == 3:
                        result = {
                            "timestamp": timestamp,
                            "estimatorResult": estimator_result,
                        }
                    else:
                        raise SyntaxError(
                            f"Detail level ({detail_level}) must be an integer between 0 and 3 (inclusive)."
                        )
                    threats.append(result)
                except EstimatorError:
                    pass

        return {
            "protocol": str(protocol) + "-" + str(key_size),
            "threats": threats,
        }

    def generate_report(
        self,
        protocols: list[dict],
        detail_level: int = 1,
    ) -> list:
        """Predict the threats against several cryptographic protocols.

        Parameters
        ----------
        protocols : list[dict]
            List of cryptographic protocols.
        detail_level : int, optional
            Level of detail in the output, by default 1.

        Returns
        -------
        list
            Threat report.
        """
        threat_report = []
        for protocol_and_key_size in protocols:
            threat_report.append(
                self.estimate_threats(
                    protocol_and_key_size["algorithm"],
                    protocol_and_key_size["keySize"],
                    detail_level=detail_level,
                )
            )

        self._threat_report = threat_report
        return threat_report

    def save_report(self, file_name: str, file_path: Path = None) -> None:
        """Save the threat report as a JSON file.

        Parameters
        ----------
        file_name : str
            File name.
        file_path : Path, optional
            File path. If unspecified, the file will be saved to the same directory
            the function is executed from.

        Raises
        ------
        AttributeError
            If the threat report has not yet been generated.
        """
        if file_path is None:
            file_path = str(Path(__file__).resolve().parent)
        try:
            with Path.open(file_path + "/" + file_name + ".json", "w") as fp:
                json.dump(self._threat_report, fp, indent=4)
        except AttributeError:
            raise AttributeError(
                "The threat report has not been generated and thus cannot be saved."
            )
