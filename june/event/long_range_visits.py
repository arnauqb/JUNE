from typing import List, Dict, Union
from random import random, shuffle, randint
import logging
import datetime

from .event import Event
from june.utils import parse_age_probabilities
from june.world import World
from june.geography import SuperArea
from june.groups import ExternalGroup, ExternalSubgroup

from june.mpi_setup import mpi_rank, mpi_size, MovablePeople, export_import_info

logger = logging.getLogger("long_range_visits")

class LongRangeDestination:
    def __init__(self, hid, rank):
        self.id = hid
        self.rank = rank

    def __eq__(self, other: "LongRangeDestination"):
        # add if type(other) !=raise TypeError in here?
        if self.id == other.id and self.rank == other.rank:
            return True
        return False

    def __ne__(self, other: "LongRangeDestination"):
        return not self.__eq__(other)


class LongRangeVisits(Event):
    def __init__(
        self,
        start_time: Union[str, datetime.datetime],
        end_time: Union[str, datetime.datetime],
        household_travel_probabilities: Dict[str, float],
        households_with_individual_destination: List,
        general_travel_probability: float=0.3,
        d0: float=50 # km
    ):
        super().__init__(start_time=start_time, end_time=end_time)
        self.household_travel_probabilities = household_travel_probabilities
        self.households_with_individual_destinations = households_with_individual_destination
        self.general_travel_probability = general_travel_probability
        self.d0 = d0
        self.leavers = [] # We don't want to be able to visit households who are already visiting other people.

    def initialise(self, world: World):
        self._link_households_long_range(world=world)

    def apply(self, world: World, activities, is_weekend: bool):
        print("applying LRVs!")
        if (
            "leisure" not in activities
        ):
            return
        to_send_abroad = MovablePeople()
        busy_abroad = {rank:[] for rank in range(mpi_size)}
        for household in world.households:
            destination = None
            external_group = None
            household_to_visit = None
            for person in household.residents:
                if not person.available:
                    continue
                if person.long_range_destination is None:
                    continue
                if person.long_range_destination.rank == mpi_rank: # if internal
                    if destination is None or destination != person.long_range_destination:
                        destination = person.long_range_destination
                        household_to_visit = world.households.get_from_id(person.long_range_destination.id)
                        #for visited_person in household_to_visit.residents:
                        #    visited_person.busy = True # only really need to do this once.
                    household_to_visit.add(person, activity="leisure")
                    for visited_person in household_to_visit.residents:
                        visited_person.residence.append(person)

                else:
                    if external_group is None or destination != person.long_range_destination:
                        long_range_destination = person.long_range_destination
                        external_group = ExternalGroup(
                            long_range_destination.id, "household", long_range_destination.rank
                        )
                    external_subgroup = ExternalSubgroup(external_group, person.residence)
                    to_send_abroad.add_person(person, external_subgroup)
                    person.busy = True

        if mpi_size > 1:
            # find out who's busy.        
            busy_internal = export_import_info(busy_abroad, collate=True)
            # tell internal people that they are busy; someone is visiting them.
            for rank, busy_households in busy_internal.items():
                for household_id in busy_households:
                    for person in world.households.get_from_id[household_id].residents:
                        person.busy = True
        return to_send_abroad

    def _link_households_long_range(self, world: World):
        internal_visits, to_visit_external = self._allocate_internal_external_visits(world)
        external_visits = self._export_import_visit_info(to_visit_external, world)
        # handle internals first
        self._assign_long_range_destinations(internal_visits, mpi_rank, world)
        # now assign destination for external travellers.
        if external_visits is not None:
            for rank, rank_visits in external_visits.items():
                if rank == mpi_rank:
                    print(rank_visits)
                    assert len(rank_visits) == 0
                    continue
                self._assign_long_range_destinations(rank_visits, rank, world)

    def _allocate_internal_external_visits(self, world: World):
        """
        Ultimately want to get a LongRangeDestination that each person will visit.
        Can do this immediately for internal visits: know all of "interal" household.ids.
        For external, just store which super_areas each person will visit for now - sort
        household ids in the _export_import step with a little bit of MPI.
        """
        internal_visits = {}
        to_visit_external = {rank: {} for rank in range(mpi_size)}

        total_not_leaving_students = 0
        for household in world.households:
            if household.type in self.households_with_individual_destinations:
                # each person in a student/communal house can go somewhere separate.
                total_leavers = 0
                for person in household.residents:
                    # communal can be student or old?
                    if random() < self.household_travel_probabilities.get(
                        household.type, self.general_travel_probability
                    ):
                        super_area = self._select_super_area_to_visit(household, world)
                        if super_area.external:
                            tup = (household.id, person.id) # This is quite ugly.
                            to_visit_external[super_area.domain_id][tup] = super_area.id
                            continue
                        household_to_visit = self._select_household_from_super_area(super_area)
                        internal_visits[person.id] = household_to_visit.id
                        total_leavers += 1
                    else:
                        total_not_leaving_students += 1
                if total_leavers == len(household.residents):
                    # maybe superfluous, as we don't visit communal households anyway?
                    self.leavers.append(household.id)
            else:
                # "normal" households/families will probably all travel together.
                if random() < self.household_travel_probabilities.get(
                    household.type, self.general_travel_probability
                ):
                    super_area = self._select_super_area_to_visit(household, world)
                    if super_area.external:
                        to_visit_external[super_area.domain_id][(household.id,)] = super_area.id
                        continue
                    household_to_visit = self._select_household_from_super_area(super_area)
                    for person in household.residents:
                        internal_visits[person.id] = household_to_visit.id
                    self.leavers.append(household.id)
        return internal_visits, to_visit_external

    def _export_import_visit_info(self, to_visit_external, world: World):
        """
        The MPI bit for initialising.
        """
        if mpi_size == 1:
            return None
        # send info about where people will go, bring in info about who is visiting this domain.
        incoming_visitor_households = export_import_info(to_visit_external)
        print("INCOMING", incoming_visitor_households)
        households_for_incoming_visitors = {rank: {} for rank in range(mpi_size)}
        # choose a household for each of them to visit here
        for rank, visitor_dict in incoming_visitor_households.items():
            print("RANK IS", rank, visitor_dict)
            if rank == mpi_rank:
                assert len(visitor_dict) == 0
                continue
            for tup, super_area_id in visitor_dict.items():
                super_area = world.super_areas.get_from_id(super_area_id)
                household = self._select_household_in_super_area(super_area)
                households_for_incoming_visitors[rank][tup] = household.id

        # send it all back again (and vv.)
        external_households_to_visit = export_import_info(households_for_incoming_visitors)

        print(external_households_to_visit)

        # sort it nicely on a person-by-person basis.
        external_visits = {rank: {} for rank in range(mpi_size)}
        for rank, external_data in external_households_to_visit.items():
            for tup, household_to_visit_id in external_data:
                if len(tup) == 1:
                    household = world.households.get_from_id(tup[0])
                    for person in household.residents:
                        external_visits[rank][person.id] = household_to_visit_id
                elif len(tup) == 2:
                    external_visits[rank][tup[1]] = household_to_visit_id
                else:
                    raise ValueError(
                        f"Tuple {tup} describing a visit to an external household should be len 1 or 2."
                    )

        print("LEN EXTERNAL VISITS", len(external_visits))
        return external_visits

    def _assign_long_range_destinations(self, visits, rank_to_visit, world: World):
        for person in world.people:
            destination_id = visits.get(person.id, None)
            if destination_id is not None:
                person.long_range_destination = LongRangeDestination(destination_id, rank_to_visit)

    def _select_super_area_to_visit(self, household, world: World):
        super_area = household.area.super_area
        while super_area.id == household.area.super_area.id:
            # it's boring to visit your own super_area, pick a new one!
            super_area = self._select_random_uniform_super_area(world)
        return super_area

    def _select_random_uniform_super_area(self, world: World):
        return world.super_areas[randint(0, len(world.super_areas)-1)]

    def _select_household_from_super_area(self, super_area: SuperArea):
        household = None
        areas = list(super_area.areas)
        shuffle(areas)
        for area in areas:
            household = self._select_household_from_area(area)
            if household is not None:
                return household
        return household

    def _select_household_from_area(self, area, max_attempts=100):
        household = None
        attempts = 0
        while attempts < max_attempts:
            trial_household = area.households[randint(0, len(area.households)-1)]
            if trial_household.type not in ["student", "communal"] and trial_household.id not in self.leavers:
                household = trial_household
                return household
        logger.warn(f"No household to visit found for rank {mpi_rank}")
        return household