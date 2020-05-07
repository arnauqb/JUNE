import logging
import logging.config
import os
import pickle

import numpy as np
import yaml
from tqdm.auto import tqdm  # for a fancy progress bar

from covid import interaction
from covid.box_generator import BoxGenerator
from covid.commute import CommuteGenerator
from covid.groups import *
from covid.groups.people import HealthIndex
from covid.infection import Infection
from covid.infection import symptoms
from covid.infection import transmission
from covid.inputs import Inputs
from covid.logger import Logger
from covid.time import Timer

world_logger = logging.getLogger(__name__)


class World:
    """
    Stores global information about the simulation
    """

    def __init__(
            self, config_file=None, box_mode=False, box_n_people=None, box_region=None
    ):
        world_logger.info("Initializing world...")
        # read configs
        self.config = read_config(config_file)
        self.relevant_groups = self.get_simulation_groups()
        self.read_defaults()
        # set up logging
        self.world_creation_logger(self.config["logger"]["save_path"])
        # start initialization
        self.box_mode = box_mode
        self.people = []
        self.total_people = 0
        self.inputs = Inputs(zone=self.config["world"]["zone"])
        if self.box_mode:
            self.initialize_hospitals()
            # self.initialize_cemeteries()
            self.initialize_box_mode(box_region, box_n_people)
        else:
            self.commute_generator = CommuteGenerator.from_file(
                self.inputs.commute_generator_path
            )
            self.initialize_areas()
            self.initialize_super_areas()
            self.initialize_people()
            self.initialize_households()
            self.initialize_hospitals()
            self.initialize_cemeteries()
            if "schools" in self.relevant_groups:
                self.initialize_schools()
            else:
                world_logger.info("schools not needed, skipping...")
            if "companies" in self.relevant_groups:
                self.initialize_companies()
            else:
                world_logger.info("companies not needed, skipping...")
            if "boundary" in self.relevant_groups:
                self.initialize_boundary()
            else:
                world_logger.info("nothing exists outside the simulated region")
            if "pubs" in self.relevant_groups:
                self.initialize_pubs()
                self.group_maker = GroupMaker(self)
            else:
                world_logger.info("pubs not needed, skipping...")
        #self.interaction = self.initialize_interaction()
        #self.logger = Logger(
        #    self, self.config["logger"]["save_path"], box_mode=box_mode
        #)
        world_logger.info("Done.")

    def world_creation_logger(
            self, save_path, config_file=None, default_level=logging.INFO,
    ):
        """
        """
        # where to read and write files
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "configs",
                "config_world_creation_logger.yaml",
            )
        # creating logger
        log_file = os.path.join(save_path, "world_creation.log")
        if os.path.isfile(config_file):
            with open(config_file, "rt") as f:
                log_config = yaml.safe_load(f.read())
            logging.config.dictConfig(log_config)
        else:
            logging.basicConfig(filename=log_file, level=logging.INFO)

    def to_pickle(self, pickle_obj=os.path.join("..", "data", "world.pkl")):
        """
        Write the world to file. Comes in handy when setting up the world
        takes a long time.
        """
        with open(pickle_obj, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, pickle_obj="/cosma7/data/dp004/dc-quer1/world.pkl"):
        """
        Initializes a world instance from an already populated world.
        """
        with open(pickle_obj, "rb") as f:
            world = pickle.load(f)
        return world

    @classmethod
    def from_config(cls, config_file):
        return cls(config_file)

    def get_simulation_groups(self):
        """
        Reads all the different groups specified in the time section of the configuration file.
        """
        timesteps_config = self.config["time"]["step_active_groups"]
        active_groups = []
        for daytype in timesteps_config.keys():
            for timestep in timesteps_config[daytype].values():
                for group in timestep:
                    active_groups.append(group)
        active_groups = np.unique(active_groups)
        return active_groups

    def read_defaults(self):
        """
        Read config files for the world and it's active groups.
        """
        basepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "configs", "defaults",
        )
        # global world settings
        default_config_path = os.path.join(basepath, "world.yaml")
        with open(default_config_path, "r") as f:
            default_config = yaml.load(f, Loader=yaml.FullLoader)
        for key in default_config.keys():
            if key not in self.config:
                self.config[key] = default_config[key]
        # active group settings
        for relevant_group in self.relevant_groups:
            group_config_path = os.path.join(basepath, f"{relevant_group}.yaml")
            if os.path.isfile(group_config_path):
                with open(group_config_path, "r") as f:
                    default_config = yaml.load(f, Loader=yaml.FullLoader)
                for key in default_config.keys():
                    if key not in self.config:
                        self.config[key] = default_config[key]

    def initialize_box_mode(self, region=None, n_people=None):
        """
        Sets the simulation to run in a single box, with everyone inside and no 
        schools, households, etc.
        Useful for testing interaction models and comparing to SIR.
        """
        world_logger.info("Setting up box mode...")
        self.boxes = Boxes()
        box = BoxGenerator(self, region, n_people)
        self.boxes.members = [box]
        self.people = People(self)
        self.people.members = box.people

    def initialize_cemeteries(self):
        self.cemeteries = Cemeteries(self)

    def initialize_hospitals(self):
        self.hospitals = Hospitals.from_file(
            self.inputs.hospital_data_path,
            self.inputs.hospital_config_path,
            box_mode=self.box_mode,
        )

        if not self.box_mode:
            pbar = tqdm(total=len(self.super_areas.members))
            for msoarea in self.super_areas.members:
                distributor = HospitalDistributor(self.hospitals, msoarea)
                pbar.update(1)
            pbar.close()

    def initialize_pubs(self):
        self.pubs = Pubs(self, self.inputs.pubs_df, self.box_mode)

    def initialize_areas(self):
        """
        Each output area in the world is represented by an Area object.
        This Area object contains the demographic information about
        people living in it.
        """
        print("Initializing areas...")
        self.areas = Areas.from_file(
            self.inputs.n_residents_file,
            self.inputs.age_freq_file,
            self.inputs.sex_freq_file,
            self.inputs.household_composition_freq_file,
        )
        areas_distributor = AreaDistributor(
            self.areas,
            self.inputs.area_mapping_df,
            self.inputs.areas_coordinates_df,
            self.relevant_groups,
        )
        areas_distributor.read_areas_census()

    def initialize_super_areas(self):
        """
        An super_area is a group of areas. We use them to store company data.
        """
        print("Initializing SuperAreas...")
        self.super_areas = SuperAreas(self)
        super_areas_distributor = SuperAreaDistributor(
            self.super_areas,
            self.relevant_groups,
        )

    def initialize_people(self):
        """
        Populates the world with person instances.
        """
        print("Initializing people...")
        # TODO:
        # self.people = People.from_file(
        #    self.inputs.,
        #    self.inputs.,
        # )
        self.people = People(self)
        pbar = tqdm(total=len(self.areas.members))
        for area in self.areas.members:
            # get msoa flow data for this oa area
            wf_area_df = self.inputs.workflow_df.loc[(area.super_area.name,)]
            person_distributor = PersonDistributor(
                self,
                self.people,
                self.areas,
                area,
                self.super_areas,
                self.inputs.compsec_by_sex_df,
                wf_area_df,
                self.inputs.key_compsec_ratio_by_sex_df,
                self.inputs.key_compsec_distr_by_sex_df,
            )
            person_distributor.populate_area()
            pbar.update(1)
        pbar.close()

    def initialize_households(self):
        """
        Calls the HouseholdDistributor to assign people to households following
        the census household compositions.
        """
        print("Initializing households...")
        pbar = tqdm(total=len(self.areas.members))
        self.households = Households(self)
        for area in self.areas.members:
            household_distributor = HouseholdDistributor(
                self.households,
                area,
                self.areas,
                self.config,
            )
            household_distributor.distribute_people_to_household()
            pbar.update(1)
        pbar.close()

    def initialize_schools(self):
        """
        Schools are organized in NN k-d trees by age group, so we can quickly query
        the closest age compatible school to a certain kid.
        """
        print("Initializing schools...")
        self.schools = Schools.from_file(
            self.inputs.school_data_path, self.inputs.school_config_path
        )
        pbar = tqdm(total=len(self.areas.members))
        for area in self.areas.members:
            self.distributor = SchoolDistributor.from_file(
                self.schools,
                area,
                self.inputs.school_config_path,
            )
            self.distributor.distribute_kids_to_school()
            self.distributor.distribute_teachers_to_school()
            pbar.update(1)
        pbar.close()

    def initialize_companies(self):
        """
        Companies live in super_areas.
        """
        print("Initializing Companies...")
        self.companies = Companies.from_file(
            self.inputs.companysize_file, self.inputs.company_per_sector_per_msoa_file,
        )
        pbar = tqdm(total=len(self.super_areas.members))
        for super_area in self.super_areas.members:
            self.distributor = CompanyDistributor(
                self.companies,
                super_area,
                self.config,
            )
            self.distributor.distribute_adults_to_companies()
            pbar.update(1)
        pbar.close()

    def initialize_boundary(self):
        """
        Create a population that lives in the boundary.
        It interacts with the population in the simulated region only
        in companies. No interaction takes place during leasure activities.
        """
        print("Creating Boundary...")
        self.boundary = Boundary(self)


def read_config(config_file):
    if config_file is None:
        config_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "config_example.yaml",
        )
    with open(config_file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

