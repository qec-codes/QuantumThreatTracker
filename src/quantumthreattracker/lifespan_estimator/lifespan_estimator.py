"""Class for estimating the remaining lifespan of cryptographic protocols."""

import time
from datetime import datetime
from typing import List

from quantumthreattracker.algorithms.baseline_shor import (
    BaselineShor,
    BaselineShorParams,
)


class HardwareRoadmap:
    """Class for representing quantum computing hardware roadmaps."""

    def __init__(self):
        self._hardware_roadmap = []

    def list(self) -> List:
        """Get the hardware roadmap as a list.

        Returns
        -------
        List
            Hardware roadmap; expressed as a list of timestamps, together with the
            quantum computing hardware expected to be available at that timestamp.
        """
        return self._hardware_roadmap

    def enumerate(self) -> List:
        """Get the hardware roadmap with indicies attached to each entry.

        Returns
        -------
        List
            Hardware roadmap with additional entries for indicies.
        """
        indexed_roadmap = []
        for timestamp_index in range(len(self._hardware_roadmap)):
            indexed_roadmap.append(
                {"index": timestamp_index, **self._hardware_roadmap[timestamp_index]}
            )
            for qc_index in range(len(indexed_roadmap[timestamp_index]["hardware"])):
                indexed_roadmap[timestamp_index]["hardware"][qc_index] = {
                    "index": qc_index,
                    **indexed_roadmap[timestamp_index]["hardware"][qc_index],
                }

        return indexed_roadmap

    def add(self, timestamp: int = None, year: int = None, **qc_specs):
        """Add an entry to the hardware roadmap.

        Parameters
        ----------
        timestamp : int, optional
            Unix timestamp at which we expect the quantum computer to be available.
        year : int, optional
            Year at which we expect the quantum computer to be available.

        Raises
        ------
        SyntaxError
            If neither the timestamp nor the year are given, or both are given.
        """
        if (timestamp is None and year is None) or (
            timestamp is not None and year is not None
        ):
            raise SyntaxError(
                "Exactly one of the timestamp and the year must be given."
            )

        if year is not None:
            dt = datetime(year=year, month=1, day=1)
            timestamp = time.mktime(dt.timetuple())

        new_milestone = {"timestamp": timestamp, "hardware": [qc_specs]}

        # This slightly messy chunk of code is for adding the new quantum computer to
        # the existing hardware roadmap as a new entry in the list, or combining it with
        # an existing entry if another entry with the same timestamp already exists.
        for index in range(len(self._hardware_roadmap)):
            if timestamp < self._hardware_roadmap[index]["timestamp"]:
                self._hardware_roadmap.insert(index, new_milestone)
                return
            if timestamp == self._hardware_roadmap[index]["timestamp"]:
                self._hardware_roadmap[index]["hardware"].append(
                    new_milestone["hardware"]
                )
                return
        self._hardware_roadmap.append(new_milestone)

    def remove(self, timestamp_index: int, qc_index: int = None):
        """Remove an entry from the hardware roadmap.

        Parameters
        ----------
        timestamp_index : int
            Timestamp index
        qc_index : int, optional
            Quantum computer index. If unspecified, removes all entries in the given
            timestamp.
        """
        if qc_index is None:
            self._hardware_roadmap.pop(timestamp_index)
        else:
            self._hardware_roadmap[timestamp_index]["hardware"].pop(qc_index)


class LifespanEstimator:
    """Class for estimating the remaining lifespan of cryptographic protocols."""

    def __init__(self, hardware_roadmap: HardwareRoadmap):
        self._hardware_roadmap = hardware_roadmap

    def estimate_lifespan(
        self, protocol: str, key_size: int, formatted: bool = False
    ) -> int | datetime | None:
        """Estimate the remaining lifespan of a cryptographic protocol.

        Parameters
        ----------
        protocol : str
            Cryptographic protocol.
        key_size : int
            Key size.
        formatted : bool, optional
            If True, returns a UTC date rather than a unix timestamp.

        Returns
        -------
        int | datetime | None
            Time in the future at which we expect the protocol to be broken.
            Returns None if there is no point in the hardware roadmap at which we have
            the requisite hardware.
        """
        algorithm_params = BaselineShorParams(protocol=protocol, key_size=key_size)
        algorithm = BaselineShor(algorithm_params=algorithm_params)

        for milestone in self._hardware_roadmap.list():
            for quantum_computer in milestone["hardware"]:
                gate_error_rate = quantum_computer["gate_error_rate"]

                azure_estimate = algorithm.estimate_resources_azure(
                    {
                        "qubitParams": {
                            "name": "qubit_gate_ns_e3",
                            "oneQubitMeasurementErrorRate": gate_error_rate,
                            "oneQubitGateErrorRate": gate_error_rate,
                            "twoQubitGateErrorRate": gate_error_rate,
                            "tGateErrorRate": gate_error_rate,
                        }
                    }
                )

                num_qubits_needed = azure_estimate["physicalCounts"]["physicalQubits"]
                num_qubits_available = quantum_computer["n_qubits"]

                if num_qubits_available >= num_qubits_needed:
                    timestamp = milestone["timestamp"]
                    if formatted:
                        return datetime.fromtimestamp(timestamp).date()
                    return timestamp

        return None
