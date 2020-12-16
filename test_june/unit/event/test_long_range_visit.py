import numpy as np
import pytest

from june.event import Event, Events, LongRangeVisits, LongRangeDestination
from june.world import World
from june.groups import Household, Households
from june.geography import Area, Areas, SuperArea, SuperAreas, Region, ExternalSuperArea
from june.demography import Population, Person

from june.simulator import Simulator

def reset_all_destinations(world):
    for person in world.people:
        person.long_range_destination = None



def test__long_range_destination():
    dest = LongRangeDestination(101, 0)
    assert dest.id == 101
    assert dest.rank == 0

    dest1 = LongRangeDestination(1001, 0)
    dest2 = LongRangeDestination(1001, 1)
    dest3 = LongRangeDestination(1001, 1)
    assert dest1 != dest2
    assert dest2 == dest3
    assert dest2 is not dest3 # demonstrate behaviour; different object.

    dest4 = dest3
    assert dest3 == dest4
    assert dest3 is dest4 # shallow copy

#class TestLongRangeVisits:
@pytest.fixture(name="world")
def make_world():
    world = World()
    region = Region()
    super_areas = []
    areas = []
    households = []
    people = []
    for _ in range(10):
        _areas = []
        for _ in range(5):
            area = Area()
            area.households = []
            areas.append(area)
            _areas.append(area)
            for _ in range(5):
                household = Household(type="old", area=area)
                p1 = Person.from_attributes(age=80)
                p2 = Person.from_attributes(age=75)
                household.add(p1)
                people.append(p1)
                people.append(p2)
                household.add(p2)
                households.append(household)
                area.households.append(household)
            for _ in range(4):
                household = Household(type="student", area=area)
                p1 = Person.from_attributes(age=19)
                p2 = Person.from_attributes(age=22)
                p3 = Person.from_attributes(age=22)
                household.add(p1)
                household.add(p2)
                household.add(p3)
                people.append(p1)
                people.append(p2)
                people.append(p3)
                area.households.append(household)
                households.append(household)
            for _ in range(20):
                household = Household(type="family", area=area)
                p1 = Person.from_attributes(age=50)
                p2 = Person.from_attributes(age=48)
                p3 = Person.from_attributes(age=15)
                household.add(p1)
                household.add(p2)
                household.add(p3)
                people.append(p1)
                people.append(p2)
                people.append(p3)
                area.households.append(household)
                households.append(household)
        super_area = SuperArea(areas=_areas, region=region)
        for area in _areas:
            area.super_area = super_area
        super_areas.append(super_area)
    world.areas = Areas(areas, ball_tree=False)
    world.super_areas = SuperAreas(super_areas, ball_tree=False)
    world.households = Households(households)
    world.people = Population(people)
    for person in world.people:
        person.busy = False
        person.subgroups.leisure = None
    for household in world.households:
        household.clear()
    return world

@pytest.fixture(name="household_travel_probabilities")
def make_probs():
    household_travel_probabilities = {"family": 0.3, "student": 0.9, "old": 0.0}
    return household_travel_probabilities

@pytest.fixture(name="households_with_individual_destination")
def make_indiv_destinations():
    households_with_individual_destination = ["student", "communal"]
    return households_with_individual_destination

@pytest.fixture(name="long_range_visits")
def make_long_range_visits_event(
    #self, 
    household_travel_probabilities, 
    households_with_individual_destination,
    world
):
    long_range_visits = LongRangeVisits(
        start_time="1900-01-01",
        end_time="2999-01-01",
        household_travel_probabilities=household_travel_probabilities,
        households_with_individual_destination=households_with_individual_destination,
    )
    long_range_visits.initialise(world=world)
    return long_range_visits

def test__travel_probs_read(long_range_visits):
    assert long_range_visits.household_travel_probabilities["family"] == 0.3
    assert long_range_visits.household_travel_probabilities["old"] == 0.0
    assert long_range_visits.household_travel_probabilities["student"] == 0.9

def test__select_super_area(long_range_visits, world):
    for household in world.households:
        super_area = long_range_visits._select_super_area_to_visit(household, world)
        assert super_area.id != household.area.super_area.id

def test__allocation(long_range_visits, world):
    internal_visits, to_visit_external = long_range_visits._allocate_internal_external_visits(world)

    assert len(to_visit_external) is None # no mpi in this test...



def test__household_linking(long_range_visits, world):
    reset_all_destinations(world)
    long_range_visits.initialise(world)

    total_students = sum([len(h.residents) for h in world.households if h.type == "student"])
    total_student_travellers = 0
    unique_student_lrds = []

    unique_family_lrds = 0
    total_family_households = 0
    total_families_travelling = 0

    for household in world.households:
        # do some tests for people who travel to individual locations (ie, students here)
        if household.type == "student":
            unique_lrds = []
            student_travellers = 0
            for person in household.residents:
                lrd = person.long_range_destination
                if lrd is None:
                    continue
                if lrd not in unique_lrds:
                    unique_student_lrds.append(lrd)
                student_travellers += 1
            total_student_travellers += student_travellers

        elif household.type == "family":
            lrd = household.residents[0].long_range_destination
            total_family_households += 1
            if lrd is not None:
                total_families_travelling += 1                
            for person in household.residents:
                if lrd is None:
                    assert person.long_range_destination is None
                else:
                    assert person.long_range_destination == lrd

        elif household.type == "old":
            for person in household.residents:
                assert person.long_range_destination is None
    # 0.8 instead of 0.9?? allow for some students to pick the same place...
    assert len(unique_student_lrds) / total_student_travellers > 0.8 
    assert 0.85 < total_student_travellers / total_students < 0.95
    assert 0.25 < total_families_travelling / total_family_households < 0.35

def test__apply_long_range_visits(long_range_visits, world):
    
    pass