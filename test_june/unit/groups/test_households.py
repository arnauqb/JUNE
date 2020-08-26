import pandas as pd
import pytest
import numpy as np
from june.demography.geography import Area, Geography
from june.groups.household import Household, Households

#    HouseholdSingle,
#    HouseholdCouple,
#    HouseholdFamily,
#    HouseholdStudent,
#    HouseholdCommunal,
# )
from june.demography import Person


def test__household_class():
    household = Household(area="test", size=3)
    assert household.area == "test"
    assert household.max_size == 3


def test__households_from_sizes():
    geography = Geography.from_file({"area": ["E00000024"]})
    households = Households.for_geography(geography)
    assert len(households) == 147 + 6  # households and communal
    sizes_num = [86, 44, 10, 4, 3, 0, 0, 0]
    for i in range(1, 9):
        size_num = sizes_num[i - 1]
        ll = [
            household
            for household in households
            if (household.max_size == i) and (household.type != "communal")
        ]
        assert len(ll) == size_num

    # test communal
    communal_sizes = [
        household.max_size for household in households if household.type == "communal"
    ]
    for size in communal_sizes:
        assert np.isclose(size, 41 / 6, atol=1)


# class TestHousehold:
#    def test__add_leisure(self):
#        household = Household()
#        p = Person.from_attributes(sex="f", age=8)
#        household.add(p, activity="leisure")
#        assert p not in household.residents
#        assert p in household.kids
#        assert household.n_kids == 1
#        p = Person.from_attributes(sex="f", age=18)
#        household.add(p, activity="leisure")
#        assert p in household.young_adults
#        assert p not in household.residents
#        assert household.n_young_adults == 1
#        p = Person.from_attributes(sex="f", age=48)
#        household.add(p, activity="leisure")
#        assert p in household.adults
#        assert p not in household.residents
#        assert household.n_adults == 1
#        p = Person.from_attributes(sex="f", age=78)
#        household.add(p, activity="leisure")
#        assert p in household.old_adults
#        assert p not in household.residents
#        assert household.size == 4
#        assert household.n_old_adults == 1
#
#    def test__add_resident(self):
#        household = Household()
#        p = Person.from_attributes(sex="f", age=8)
#        household.add(p, activity="residence")
#        assert p in household.residents
#        assert p in household.kids
#        p = Person.from_attributes(sex="f", age=18)
#        household.add(p, activity="residence")
#        assert p in household.young_adults
#        assert p in household.residents
#        p = Person.from_attributes(sex="f", age=48)
#        household.add(p, activity="residence")
#        assert p in household.adults
#        assert p in household.residents
#        p = Person.from_attributes(sex="f", age=78)
#        household.add(p, activity="residence")
#        assert p in household.old_adults
#        assert p in household.residents
#
#
# class TestHouseholdSingle:
#    def test__household_single(self):
#        household = HouseholdSingle(area="test", old=False)
#        assert household.composition.n_young_adults_range == (0,1)
#        assert household.composition.n_kids_range == (0, 0)
#        assert household.composition.n_adults_range == (0, 1)
#        assert household.spec == "household"
#        assert household.type == "single"
#        assert household.area == "test"
#        assert household.max_size == 1
#        household = HouseholdSingle(area="test", old=True)
#        assert household.composition.n_kids_range == (0, 0)
#        assert household.composition.n_young_adults_range == (0, 0)
#        assert household.composition.n_adults_range == (0, 0)
#        assert household.composition.n_old_adults_range == (1, 1)
#        assert household.max_size == 1
#
#
# class TestHouseholdCouple:
#    def test__household_couple(self):
#        household = HouseholdCouple(old=False)
#        assert household.composition.n_kids_range == (0,0)
#        assert household.composition.n_young_adults_range == (1,2)
#        assert household.composition.n_adults_range == (1,2)
#        assert household.composition.n_old_adults_range == (0,1)
#        assert household.type == "couple"
#        assert household.spec == "household"
#        assert household.max_size == 2
#        household = HouseholdCouple(old=True)
#        assert household.composition.n_adults_range == (0,0)
#        assert household.composition.n_old_adults_range == (2,2)
#        assert household.composition.n_young_adults_range == (0,0)
#        assert household.composition.n_kids_range == (0,0)
#
#
# class TestHouseholdFamily:
#    def test__household_family(self):
#        household = HouseholdFamily(n_parents=1, n_kids_range=(2,np.inf))
#        assert household.composition.n_kids_range == (2, np.inf)
#        assert household.composition.n_adults_range == (0,1)
#        assert household.composition.n_young_adults_range == (0,1)
#        assert household.composition.n_old_adults_range == (0,0)
#        assert household.type == "family"
#        assert household.spec == "household"
#        assert household.max_size == np.inf
#        household = HouseholdFamily(n_parents=2, n_young_adults_range=(1,1), n_kids_range=1, n_old_adults_range=(1,2))
#        assert household.composition.n_kids_range == (1,1)
#        assert household.composition.n_adults_range == (2,2)
#        assert household.composition.n_young_adults_range == (1,1)
#        assert household.composition.n_old_adults_range == (1,2)
#
#
# class TestHouseholdStudent:
#    def test__household_student(self):
#        household = HouseholdStudent(n_students=4)
#        assert household.type == "student"
#        assert household.spec == "household"
#        assert household.composition.n_young_adults_range == (4, 4)
#
#
# class TestHouseholdCommunal:
#    def test__household_student(self):
#        household = HouseholdCommunal()
#        assert household.type == "communal"
#        assert household.spec == "household"
#        assert household.composition.n_kids_range == (0, np.inf)
#        assert household.composition.n_young_adults_range == (0, np.inf)
#        assert household.composition.n_adults_range == (0, np.inf)
#        assert household.composition.n_old_adults_range == (0, np.inf)
#
#
# class TestHouseholdsCreation:
#    @pytest.fixture(name="households", scope="module")
#    def create_households(self):
#        compositions = {
#            "single": {1: {"old": True, "number": 10}, 2: {"old": False, "number": 5}},
#            "couple": {1: {"old": True, "number": 15}, 2: {"old": False, "number": 7}},
#            "family": {
#                1: {
#                    "n_kids": 1,
#                    "n_parents" : 2,
#                    "number": 5,
#                },
#                2: {
#                    "n_kids": "2+",
#                    "n_parents": 2,
#                    "number": 4,
#                },
#                3: {
#                    "n_kids": 1,
#                    "n_parents": 2,
#                    "n_old_adults": 1,
#                    "number": 3,
#                },
#                4: {
#                    "n_kids": 0,
#                    "n_young_adults": 2,
#                    "n_parents": 2,
#                    "n_old_adults": 0,
#                    "number": 2,
#                },
#            },
#            "student": {"number": 10},
#            "communal": {"number": 3},
#            "other": {
#                "n_kids": 0,
#                "n_young_adults": "2+",
#                "n_adults": "1+",
#                "n_old_adults": 1,
#                "number": 2,
#            },
#        }
#        households = Households.from_household_compositions(compositions)
#        return households
#
#    def test__number_of_households(self, households):
#        assert len(households) == 10 + 5 + 15 + 7 + 5 + 4 + 3 + 2 + 10 + 3 + 2
#
#    def test__households_from_compositions(self, households):
#        family_households = [
#            household for household in households if household.type == "family"
#        ]
#        assert len(family_households) == 5 + 4 + 3 + 2
#        kids = np.zeros(10)
#        young_adults = np.zeros(10)
#        adults = np.zeros(10)
#        old_adults = np.zeros(10)
#        for household in family_households:
#            if household.composition.n_kids_range == (0, 0):
#                kids[0] += 1
#            elif household.composition.n_kids_range == (1, 1):
#                kids[1] += 1
#            else:
#                kids[2] += 1
#            if household.composition.n_adults_range == (1, 1):
#                adults[1] += 1
#            elif household.composition.n_adults_range == (2, 2):
#                adults[2] += 1
#
#            if household.composition.n_old_adults_range == (0, 0):
#                old_adults[1] += 1
#
#        assert kids[0] == 2
#        assert kids[1] == 8
#        assert kids[2] == 4
#
#        assert adults[1] == 5
#        assert adults[2] == 5
#
#        assert old_adults[0] == 11
#        assert old_adults[1] == 3
#
#
#
