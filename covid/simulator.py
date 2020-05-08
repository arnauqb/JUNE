import yaml
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from covid.time import Timer
from covid import interaction
from covid.infection import Infection

default_config_filename = Path(__file__).parent.parent / "configs/config_example.yaml"


sim_logger = logging.getLogger(__name__)

#TODO: Split the config into more manageable parts for tests
class Simulator:
    def __init__(self, world: "World", interaction: "Interaction", infection: Infection, config: dict):
        """
        Class to run an epidemic spread simulation on the world

        Parameters
        ----------
        world: 
            instance of World class
        interaction:
            instance of Interaction class, determines
             
        config:
            dictionary with configuration to set up the simulation
        """
        self.world = world
        self.interaction = interaction
        self.infection = infection
        self.permanent_group_hierarchy = ['boxes', 'hospitals', 'companies', 'schools', 'carehomes', 'households']
        self.randomly_order_groups = ['pubs', 'churches',]
        self.check_inputs(config)
        self.timer = Timer(config)
       
    @classmethod
    def from_file(
        cls, world, interaction, infection, config_filename=default_config_filename
    ) -> "Simulator":

        """
        Load config for simulator from world.yaml

        Parameters
        ----------
        config_filename
            The path to the world yaml configuration

        Returns
        -------
        A Simulator
        """
        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return Simulator(world, interaction, infection, config["time"])

    def check_inputs(self, config: dict):

        # Sadly, days only have 24 hours
        assert sum(config["step_duration"]["weekday"].values()) == 24
        # even during the weekend :(
        assert sum(config["step_duration"]["weekend"].values()) == 24

        # Check that all groups given in config file are in the valid group hierarchy
        all_groups = self.permanent_group_hierarchy + self.randomly_order_groups
        for step, active_groups in config['step_active_groups']['weekday'].items():
            assert all(group in all_groups for group in active_groups) 

        for step, active_groups in config['step_active_groups']['weekend'].items():
            assert all(group in all_groups for group in active_groups) 

    def apply_group_hierarchy(self, active_groups: List[str]) -> List[str]:
        """
        Returns a list of active groups with the right order, obeying the permanent group hierarcy
        and shuflling the random one. It is very important having carehomes and households at the very end.

        Parameters
        ----------
        active_groups:
            list of groups that are active at a given time step
        Returns
        -------
        Ordered list of active groups according to hierarchy
        """
        random.shuffle(self.randomly_order_groups)
        group_hierarchy = [group for group in self.permanent_group_hierarchy if group not in ['carehomes', 'households']]
        group_hierarchy += self.randomly_order_groups + ['carehomes', 'households']
        active_groups.sort(key=lambda x: group_hierarchy.index(x))
        return active_groups

    def set_active_group_to_people(self, active_groups: List["Groups"]):
        """
        Calls the set_active_members() method of each group, if the group
        is set as active

        Parameters
        ----------
        active_groups:
            list of groups that are active at a time step
        """
        active_groups = self.apply_group_hierarchy(active_groups)
        for group_name in active_groups:
            grouptype = getattr(self.world, group_name)
            if "pubs" in active_groups:
                world.group_maker.distribute_people(group_name)
            for group in grouptype.members:
                group.set_active_members()

    def set_allpeople_free(self):
        """ 
        Set everyone's active group to None, 
        ready for next time step

        """
        for person in self.world.people.members:
            person.active_group = None

    def hospitalise_the_sick(self, person):
        """
        These functions could be more elegantly handled by an implementation inside a group collection.
        I'm putting them here for now to maintain the same functionality whilst removing a person's
        reference to the world as that makes it impossible to perform population generation prior
        to world construction.
        """
        if person.in_hospital is None:
            self.hospitals.allocate_patient(person)

    def bury_the_dead(self, person):
        cemetery = self.cemeteries.get_nearest(person)
        cemetery.add(person)
        person.household.remove_person(person)
        for group in person.groups:
            group.remove_person(person)

    def update_health_status(self, time, delta_time):

        for person in world.people.infected:
            health_information = person.health_information
            health_information.update_health_status(time, delta_time)
            if health_information.in_hospital:
                self.hospitalise_the_sick(person)
            # release patients that recovered
            elif health_information.recovered and person.hospital is not None:
                person.hospital.release_as_patient(person)
            elif health_information.must_stay_at_home:
                if person.age <= 14:
            #TODO: force set active at home for those with symptoms,
            # if <14 yo carry a parent
            # add complacency probability (for now very high)
                
            elif health_information.dead:
                self.bury_the_dead(person)

    def seed(self, group: "Group", n_infections: int):
        """
        Randomly pick people in group to seed the infection

        Parameters
        ----------
        group:
            group instance in which to seed the infection

        n_infections:
            number of random people to infect in the given group

        """
        #TODO: add attribute susceptible to people
        sim_logger.info(f"Seeding {n_infections} infections in group {group.spec}")
        choices = np.random.choice(len(self.world.people.members), n_infections, replace=False)
        infecter_reference = self.infection
        for choice in choices:
            infecter_reference.infect_person_at_time(
                list(self.world.people.members)[choice], self.timer.now
            )
        self.update_health_status()

    def do_timestep(self):
        """
        Perform a time step in the simulation

        """
        active_groups = self.timer.active_groups()
        if not active_groups or len(active_groups) == 0:
            world_logger.info("==== do_timestep(): no active groups found. ====")
            return
        # update people (where they are according to time)
        self.set_active_group_to_people(active_groups)
        # infect people in groups
        group_instances = [getattr(self.world, group) for group in active_groups]
        n_people = 0
        for group_type in group_instances:
            for group in group_type.members:
                self.interaction.time_step(self.timer.now, self.timer.duration, group)
                n_people += len(group.people)
        for cemetery in self.cemeteries.members:
            n_people += len(cemetery.people)
        # assert conservation of people
        assert n_people  == len(world.people.members)

        self.update_health_status()
        self.set_allpeople_free()

    def run(self, n_days, save=False):
        """
        Run simulation with n_seed initial infections

        Parameters
        ----------
        n_seed:
            number of initial infections

        """
        sim_logger.info(
            f"Starting group_dynamics for {self.timer.total_days} days at day {self.timer.day}"
        )
        sim_logger.info(
            f"starting the loop ..., at {self.timer.day} days, to run for {self.timer.total_days} days"
        )

        for day in self.timer:
            if day > self.timer.total_days:
                break
            self.logger.log_timestep(day)
            self.do_timestep(self.timer)
        # Save the world
        if save:
            self.world.to_pickle()
