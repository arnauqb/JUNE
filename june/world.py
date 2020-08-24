import logging
import h5py
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import Optional
from june.groups import Group
from june.box.box_mode import Boxes, Box
from june.demography import Demography, Population
from june.demography.person import Activities, Person
from june.distributors import (
    SchoolDistributor,
    HospitalDistributor,
    HouseholdDistributor,
    CareHomeDistributor,
    WorkerDistributor,
    CompanyDistributor,
    UniversityDistributor,
)
from june.demography.geography import Geography, Areas
from june.groups import *
from june.commute import CommuteGenerator

logger = logging.getLogger(__name__)

possible_groups = [
    "households",
    "care_homes",
    "schools",
    "hospitals",
    "companies",
    "universities",
    "pubs",
    "groceries",
    "cinemas",
]


def _populate_areas(areas: Areas, demography):
    people = Population()
    for area in areas:
        area.populate(demography)
        people.extend(area.people)
    return people


class World:
    """
    This Class creates the world that will later be simulated.
    The world will be stored in pickle, but a better option needs to be found.
    
    Note: BoxMode = Demography +- Sociology - Geography
    """

    def __init__(self):
        """
        Initializes a world given a geography and a demography. For now, households are
        a special group because they require a mix of both groups (we need to fix
        this later). 
        """
        self.areas = None
        self.super_areas = None
        self.people = None
        self.households = None
        self.care_homes = None
        self.schools = None
        self.companies = None
        self.hospitals = None
        self.pubs = None
        self.groceries = None
        self.cinemas = None
        self.commutecities = None
        self.commutehubs = None
        self.cemeteries = None
        self.universities = None
        self.box_mode = False

    def distribute_people(
        self, include_households=True, include_commute=False, include_rail_travel=False
    ):
        """
        Distributes people to buildings assuming default configurations.
        """

        if (
            self.companies is not None
            or self.hospitals is not None
            or self.schools is not None
            or self.care_homes is not None
        ):
            worker_distr = WorkerDistributor.for_super_areas(
                area_names=[super_area.name for super_area in self.super_areas]
            )  # atm only for_geography()
            worker_distr.distribute(
                areas=self.areas, super_areas=self.super_areas, population=self.people
            )

        if self.care_homes is not None:
            carehome_distr = CareHomeDistributor()
            carehome_distr.populate_care_home_in_areas(self.areas)

        if include_households:
            household_distributor = HouseholdDistributor.from_file()
            self.households = household_distributor.distribute_people_and_households_to_areas(
                self.areas
            )

        if self.schools is not None:
            school_distributor = SchoolDistributor(self.schools)
            school_distributor.distribute_kids_to_school(self.areas)
            school_distributor.limit_classroom_sizes()
            school_distributor.distribute_teachers_to_schools_in_super_areas(
                self.super_areas
            )


        if self.universities is not None:
            uni_distributor = UniversityDistributor(self.universities)
            uni_distributor.distribute_students_to_universities(self.super_areas)

        if include_commute:
            self.initialise_commuting()

        if include_rail_travel:
            self.initialise_rail_travel()

        if self.hospitals is not None:
            hospital_distributor = HospitalDistributor.from_file(self.hospitals)
            hospital_distributor.distribute_medics_to_super_areas(self.super_areas)

        # Companies last because need hospital and school workers first
        if self.companies is not None:
            company_distributor = CompanyDistributor()
            company_distributor.distribute_adults_to_companies_in_super_areas(
                self.super_areas
            )

    def initialise_commuting(self):
        commute_generator = CommuteGenerator.from_file()

        for area in self.areas:
            commute_gen = commute_generator.regional_gen_from_msoarea(area.name)
            for person in area.people:
                person.mode_of_transport = commute_gen.weighted_random_choice()

        # CommuteCity
        self.commutecities = CommuteCities.for_super_areas(self.super_areas)

        self.commutecity_distributor = CommuteCityDistributor(
            self.commutecities.members, self.super_areas.members
        )
        self.commutecity_distributor.distribute_people()

        # CommuteHub
        self.commutehubs = CommuteHubs(self.commutecities)
        self.commutehubs.from_file()
        self.commutehubs.init_hubs()

        self.commutehub_distributor = CommuteHubDistributor(self.commutecities.members)
        self.commutehub_distributor.from_file()
        self.commutehub_distributor.distribute_people()

        # CommuteUnit
        self.commuteunits = CommuteUnits(self.commutehubs.members)
        self.commuteunits.init_units()

        # CommuteCityUnit
        self.commutecityunits = CommuteCityUnits(self.commutecities.members)
        self.commutecityunits.init_units()

    def initialise_rail_travel(self):

        # TravelCity
        self.travelcities = TravelCities(self.commutecities)
        self.init_cities()

        # TravelCityDistributor
        self.travelcity_distributor = TravelCityDistributor(
            self.travelcities.members, self.super_areas.members
        )
        self.travelcity_distributor.distribute_msoas()

        # TravelUnit
        self.travelunits = TravelUnits()

    def to_hdf5(self, file_path: str, chunk_size=100000):
        """
        Saves the world to an hdf5 file. All supergroups and geography
        are stored as groups. Class instances are substituted by ids of the 
        instances. To load the world back, one needs to call the
        generate_world_from_hdf5 function.

        Parameters
        ----------
        file_path
            path of the hdf5 file
        chunk_size
            how many units of supergroups to process at a time.
            It is advise to keep it around 1e5
        """
        from june.hdf5_savers import save_world_to_hdf5
        save_world_to_hdf5(world=self, file_path=file_path, chunk_size=chunk_size)


def generate_world_from_geography(
    geography: Geography,
    demography: Optional[Demography] = None,
    box_mode=False,
    include_households=True,
    include_commute=False,
    include_rail_travel=False,
):
    """
        Initializes the world given a geometry. The demography is calculated
        with the default settings for that geography.
        """
    world = World()
    world.box_mode = box_mode
    if demography is None:
        demography = Demography.for_geography(geography)
    if include_rail_travel and not include_commute:
        raise ValueError("Rail travel depends on commute and so both must be true")
    if box_mode:
        world.hospitals = Hospitals.for_box_mode()
        world.people = _populate_areas(geography.areas, demography)
        world.boxes = Boxes([Box()])
        world.cemeteries = Cemeteries()
        world.boxes.members[0].set_population(world.people)
        return world
    world.areas = geography.areas
    world.super_areas = geography.super_areas
    world.people = _populate_areas(world.areas, demography)
    for possible_group in possible_groups:
        geography_group = getattr(geography, possible_group)
        if geography_group is not None:
            setattr(world, possible_group, geography_group)
    world.distribute_people(
        include_households=include_households,
        include_commute=include_commute,
        include_rail_travel=include_rail_travel,
    )
    world.cemeteries = Cemeteries()
    return world


