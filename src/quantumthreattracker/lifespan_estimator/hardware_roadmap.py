"""Class for hardware roadmaps."""

import json
from pathlib import Path
from typing import Self

from qsharp.estimator import EstimatorParams


class HardwareRoadmap:
    """Class for representing quantum computing hardware roadmaps."""

    def __init__(self, hardware_roadmap: list = None):
        if hardware_roadmap is None:
            self._hardware_roadmap = []
        else:
            self._hardware_roadmap = hardware_roadmap

    def as_list(self) -> list:
        """Get the hardware roadmap as a list.

        Returns
        -------
        list
            Hardware roadmap.
        """
        return self._hardware_roadmap

    @classmethod
    def from_file(cls, file_path: str) -> Self:
        """Import a hardware roadmap from a file.

        Parameters
        ----------
        file_path : str
            File path.

        Returns
        -------
        Self
            Hardware roadmap.
        """
        with open(file_path, "rb") as file:
            return cls(json.load(file))

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
        file_path : str, optional
            File path. If unspecified, the file will be saved to the current working
            directory.

        Raises
        ------
        AttributeError
            If the hardware roadmap has not yet been generated.
        """
        if file_path is None:
            file_path = str(Path.cwd())
        try:
            with Path.open(file_path + "/" + file_name + ".json", "w") as fp:
                json.dump(self._hardware_roadmap, fp, indent=4)
        except AttributeError:
            raise AttributeError(
                "The hardware roadmap has not been generated and thus cannot be saved."
            )
