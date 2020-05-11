import os
import yaml
import logging
from pathlib import Path
from itertools import count
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from june.groups import Group
from june.logger_creation import logger
from enum import IntEnum

from june.geography import SuperArea

logger = logging.getLogger(__name__)

default_data_filename = (
    Path(os.path.abspath(__file__)).parent.parent.parent
    / "data/processed/hospital_data/england_hospitals.csv"
)
default_config_filename = (
    Path(os.path.abspath(__file__)).parent.parent.parent
    / "configs/defaults/groups/hospitals.yaml"
)


class Hospital(Group):
    """
    The Hospital class represents a hospital and contains information about
    its patients and workers - the latter being the usual "people".

    TODO: we have to figure out the inheritance structure; I think it will
    be an admixture of household and company.
    I will also assume that the patients cannot infect anybody - this may
    become a real problem as it is manifestly not correct.

    We currently use three subgroups: 
    0 - workers (i.e. nurses, doctors, etc.),
    1 - patients
    2 - ICU patients
    """

    _id = count()

    class GroupType(IntEnum):
        workers = 0
        patients = 1
        icu_patients = 2

    __slots__ = "id", "n_beds", "n_icu_beds", "coordinates", "msoa_name"

    def __init__(
        self,
        coordinates: list,  # Optional[Tuple[float, float]] = None,
        n_beds: int,
        n_icu_beds: int,
        super_area: SuperArea = None,
    ):
        """
        Create a Hospital given its description.

        Parameters
        ----------
        hospital_id:
            unique identifier of the hospital 
        n_beds:
            total number of regular beds in the hospital
        n_icu_beds:
            total number of ICU beds in the hospital
        coordinates:
            latitude and longitude 
        msoa_name:
            name of the msoa area the hospital belongs to
        """
        self.id = next(self._id)
        super().__init__(f"Hospital_{self.id}", "hospital")
        self.super_area = super_area
        self.coordinates = coordinates
        self.n_beds = n_beds
        self.n_icu_beds = n_icu_beds

    @property
    def full(self):
        """
        Check whether all regular beds are being used
        """
        return self[self.GroupType.patients].size >= self.n_beds

    @property
    def full_ICU(self):
        """
        Check whether all ICU beds are being used
        """
        return self[self.GroupType.icu_patients].size >= self.n_icu_beds

    def set_active_members(self):
        """
        Set people in hospital active in hospital only
        """
        # TODO: We need to check what we want to do with this,
        # it will probably be taken care by the supergroup
        for person in self[self.GroupType.patients]:
            if person.active_group is None:
                person.active_group = "hospital"
        for person in self[self.GroupType.icu_patients]:
            if person.active_group is None:
                person.active_group = "hospital"

    def add(self, person, qualifier=GroupType.workers):
        super().add(person, qualifier)
        if qualifier in [self.GroupType.patients, self.GroupType.icu_patients]:
            person.in_hospital = self
        person.groups.append(self)

    def add_as_patient(self, person):
        """
        Add patient to hospital, depending on their healty information tag
        they'll go to intensive care or regular beds.

        Parameters
        ----------
        person:
            person instance to add as patient
        """
        if person.health_information.tag == "intensive care":
            self.add(person, self.GroupType.icu_patients)
        elif person.health_information.tag == "hospitalised":
            self.add(person, self.GroupType.patients)
        else:
            raise AssertionError(
                "ERROR: This person shouldn't be trying to get to a hospital"
            )

    def release_as_patient(self, person):
        """
        Release a patient from hospital

        Parameters
        ----------
        person: 
            person instance to remove as patient
        """
        for group_type in (self.GroupType.icu_patients, self.GroupType.patients):
            patient_group = self[group_type]
            if person in patient_group:
                patient_group.remove(person)
        person.in_hospital = None

    def update_status_lists_for_patients(self, time, delta_time):
        """
        Update the health information of patients, and move them around if necessary
        """
        dead = []
        patient_group = self[self.GroupType.patients]
        icu_group = self[self.GroupType.icu_patients]
        for person in patient_group.people:
            person.health_information.update_health_status(time, delta_time)
            if person.health_information.infected:
                if person.health_information.tag == "intensive care":
                    icu_group.append(person)
                    patient_group.remove(person)
            if person.health_information.recovered:
                self.release_as_patient(person)
            if person.health_information.dead:
                dead.append(person)
        for person in dead:
            patient_group.remove(person)

    def update_status_lists_for_ICUpatients(self, time, delta_time):
        """
        Update the health information of ICU patients, and move them around if necessary
        """
        patient_group = self[self.GroupType.patients]
        icu_group = self[self.GroupType.icu_patients]

        dead = []
        for person in icu_group.people:
            person.health_information.update_health_status(time, delta_time)
            if person.health_information.infected:
                if person.health_information.tag == "hospitalised":
                    patient_group.append(person)
                    icu_group.remove(person)
            if person.health_information.recovered:
                self.release_as_patient(person)
            if person.health_information.dead:
                # TODO: check what to do with dead!! bury is not there anymore
                dead.append(person)
        for person in dead:
            icu_group.remove(person)

    def update_status_lists(self, time, delta_time):
        # three copies of what happens in group for the three lists of people
        # in the hospital
        if self.contains_people:
            super().update_status_lists(time, delta_time)
            logger.info(
                f"=== update status list for hospital with {self.size} people ==="
            )
        if self[self.GroupType.patients].contains_people:
            self.update_status_lists_for_patients(time, delta_time)
            self.update_status_lists_for_ICUpatients(time, delta_time)
            logger.info(
                f"=== hospital currently has {self[self.GroupType.patients].size} "
                f"patients, and {self[self.GroupType.icu_patients].size}, ICU patients"
            )


class Hospitals:
    def __init__(
        self,
        hospitals: List["Hospital"],
        max_distance: float = 100,
        box_mode: bool = False,
    ):
        """
        Create a group of hospitals, and provide functionality to locate patients
        to a nearby hospital.

        Parameters
        ----------
        hospital_df:
            data frame with hospital data
        config:
            config dictionary
        box_mode:
            whether to run in single box mode, or full simulation
        """
        self.box_mode = box_mode
        self.max_distance = max_distance
        self.members = hospitals
        coordinates = np.array([hospital.coordinates for hospital in hospitals])
        if not box_mode:
            self.init_trees(coordinates)

    def __iter__(self):
        return iter(self.members)

    @classmethod
    def for_box_mode(cls):
        hospitals = []
        hospitals.append(Hospital(coordinates=None, n_beds=10, n_icu_beds=2,))
        hospitals.append(Hospital(coordinates=None, n_beds=5000, n_icu_beds=5000,))
        return cls(hospitals, box_mode=True)

    @classmethod
    def from_file(
        cls,
        filename: str = default_data_filename,
        config_filename: str = default_config_filename,
    ) -> "Hospitals":
        """
        Initialize Hospitals from path to data frame, and path to config file.

        Parameters
        ----------
        filename:
            path to hospital dataframe
        config_filename:
            path to hospital config dictionary

        Returns
        -------
        Hospitals instance
        """

        hospital_df = pd.read_csv(filename)
        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        max_distance = config["max_distance"]
        icu_fraction = config["icu_fraction"]
        logger.info(f"There are {len(hospital_df)} hospitals in the world.")
        hospitals = cls.init_hospitals(cls, hospital_df, icu_fraction)
        return Hospitals(hospitals, max_distance, False)

    @classmethod
    def for_geography(
        cls,
        geography,
        filename: str = default_data_filename,
        config_filename: str = default_config_filename,
    ):
        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        max_distance = config["max_distance"]
        icu_fraction = config["icu_fraction"]
        hospital_df = pd.read_csv(filename, index_col=0)
        super_area_names = [super_area.name for super_area in geography.super_areas]
        hospital_df = hospital_df.loc[super_area_names]
        logger.info(f"There are {len(hospital_df)} hospitals in this geography.")
        total_hospitals = len(hospital_df)
        hospitals = []
        hospital_counter = 0
        for super_area in geography.super_areas:
            if super_area.name in hospital_df.index:
                row = hospital_df.loc[super_area.name]
                coordinates = row[["Latitude", "Longitude"]].values.astype(np.float)
                n_beds = row["beds"]
                hospital = cls.create_hospital(
                    super_area, coordinates, n_beds, icu_fraction
                )
                hospitals.append(hospital)
                hospital_counter += 1
                if hospital_counter == total_hospitals:
                    break
        return cls(hospitals, max_distance, False)

    @classmethod
    def create_hospital(cls, super_area, coordinates, n_beds, icu_fraction):
        n_icu_beds = round(icu_fraction * n_beds)
        n_beds -= n_icu_beds
        hospital = Hospital(
            super_area=super_area,
            coordinates=coordinates,
            n_beds=n_beds,
            n_icu_beds=n_icu_beds,
        )
        return hospital

    def init_hospitals(
        self, hospital_df: pd.DataFrame, icu_fraction: float
    ) -> List["Hospital"]:
        """
        Create Hospital objects with the right characteristics,
        as given by dataframe.

        Parameters
        ----------
        hospital_df:
            dataframe with hospital characteristics data
        """
        hospitals = []
        for (index, row) in hospital_df.iterrows():
            n_beds = row["beds"]
            n_icu_beds = round(icu_fraction * n_beds)
            n_beds -= n_icu_beds
            # msoa_name = row["MSOA"]
            coordinates = row[["Latitude", "Longitude"]].values.astype(np.float)
            # create hospital
            hospital = Hospital(
                # super_area=,
                coordinates=coordinates,
                n_beds=n_beds,
                n_icu_beds=n_icu_beds,
            )
            hospitals.append(hospital)
        return hospitals

    def init_trees(self, hospital_coordinates: np.array) -> BallTree:
        """
        Reads hospital location and sizes, it initializes a KD tree on a sphere,
        to query the closest hospital to a given location.

        Parameters
        ----------
        hospital_df: 
            dataframe with hospital characteristics data

        Returns
        -------
        Tree to query nearby schools
        """
        self.hospital_trees = BallTree(
            np.deg2rad(hospital_coordinates), metric="haversine",
        )

    def allocate_patient(self, person: "Person"):
        """
        Function to allocate patients into close by hospitals with available beds.
        If there are no available beds within a maximum distance, the patient is
        not allocated.

        Parameters
        ----------
        person: 
            patient to allocate into a hospital 
        Returns
        -------
        hospital with availability

        """
        assign_icu = person.health_information.tag == "intensive care"
        assign_patient = person.health_information.tag == "hospitalised"

        if self.box_mode:
            for hospital in self.members:
                if assign_patient and not hospital.full:
                    return hospital
                if assign_icu and not hospital.full_ICU:
                    return hospital
        else:
            hospital = None
            # find hospitals  within radius of max distance
            distances, hospitals_idx = self.get_closest_hospitals(
                person.area.coordinates, self.max_distance
            )
            for distance, hospital_id in zip(distances, hospitals_idx):
                hospital = self.members[hospital_id]
                if distance > self.max_distance:
                    break
                if (assign_icu and not hospital.full) or (
                    assign_patient and not hospital.full_ICU
                ):
                    break
            if hospital is not None:
                logger.info(
                    f"Receiving hospital for patient with "
                    + f"{person.health_information.tag} at distance = {distance} km"
                )
                hospital.add_as_patient(person)
            else:
                logger.info(
                    f"no hospital found for patient with "
                    + f"{person.health_information.tag} in distance "
                    + f"< {self.max_distance} km."
                )

    def get_closest_hospitals(
        self, coordinates: Tuple[float, float], r_max: float
    ) -> Tuple[float, float]:
        """
        Get the closest hospitals to a given coordinate within r_max

        Parameters
        ----------
        coordinates: 
            latitude and longitude
        r_max:
            maximum distance to hospital

        Returns
        -------
        Distance to the closest hospitals, in km 
        ID of the hospitals within r_max, ordered by distance

        """
        earth_radius = 6371.0  # km
        r_max /= earth_radius
        idx, distances = self.hospital_trees.query_radius(
            np.deg2rad(coordinates.reshape(1, -1)),
            r=r_max,
            return_distance=True,
            sort_results=True,
        )
        distances = np.array(distances[0]) * earth_radius
        return distances, idx[0]
