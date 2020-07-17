import pandas as pd
import pytest
import numpy as np
from june.demography.geography import Area
from june.groups.household import (
    Household,
    HouseholdSingle,
    HouseholdCouple,
    HouseholdFamily,
    HouseholdStudent,
    HouseholdCommunal,
)  # Households
from june.demography import Person


class TestHousehold:
    def test__add_leisure(self):
        household = Household()
        p = Person.from_attributes(sex="f", age=8)
        household.add(p, activity="leisure")
        assert p not in household.residents
        assert p in household.kids
        assert household.n_kids == 1
        p = Person.from_attributes(sex="f", age=18)
        household.add(p, activity="leisure")
        assert p in household.young_adults
        assert p not in household.residents
        assert household.n_young_adults == 1
        p = Person.from_attributes(sex="f", age=48)
        household.add(p, activity="leisure")
        assert p in household.adults
        assert p not in household.residents
        assert household.n_adults == 1
        p = Person.from_attributes(sex="f", age=78)
        household.add(p, activity="leisure")
        assert p in household.old_adults
        assert p not in household.residents
        assert household.size == 4
        assert household.n_old_adults == 1

    def test__add_resident(self):
        household = Household()
        p = Person.from_attributes(sex="f", age=8)
        household.add(p, activity="residence")
        assert p in household.residents
        assert p in household.kids
        p = Person.from_attributes(sex="f", age=18)
        household.add(p, activity="residence")
        assert p in household.young_adults
        assert p in household.residents
        p = Person.from_attributes(sex="f", age=48)
        household.add(p, activity="residence")
        assert p in household.adults
        assert p in household.residents
        p = Person.from_attributes(sex="f", age=78)
        household.add(p, activity="residence")
        assert p in household.old_adults
        assert p in household.residents


class TestHouseholdSingle:
    def test__household_single(self):
        household = HouseholdSingle(area="test", young_adults=1)
        assert household._young_adults == 1
        assert household.spec == "household"
        assert household.type == "single"
        assert household.area == "test"
        assert household.max_size == 1


class TestHouseholdCouple:
    def test__household_couple(self):
        household = HouseholdCouple(adults=2)
        assert household._adults == 2
        assert household.type == "couple"
        assert household.spec == "household"
        assert household.max_size == 2
        household = HouseholdCouple()


class TestHouseholdFamily:
    def test__household_family(self):
        household = HouseholdFamily(kids=1, adults=2)
        assert household._kids == 1
        assert household._adults == 2
        assert household.type == "family"
        assert household.spec == "household"
        assert household.max_size == 3


class TestHouseholdStudent:
    def test__household_student(self):
        household = HouseholdStudent()
        assert household.type == "student"
        assert household.spec == "household"


class TestHouseholdCommunal:
    def test__household_student(self):
        household = HouseholdCommunal()
        assert household.type == "communal"
        assert household.spec == "household"


class TestHouseholdsCreation:
    @pytest.fixture(name="households", scope="module")
    def create_households(self):
        compositions = {
            "single": {1: {"old": True, "number": 10}, 2: {"old": False, "number": 5}},
            "couple": {1: {"old": True, "number": 15}, 2: {"old": False, "number": 7}},
            "families": {
                1: {
                    "kids": 1,
                    "young_adults": 0,
                    "adults": 1,
                    "old_adults": 0,
                    "number": 5,
                },
                2: {
                    "kids": "2+",
                    "young_adults": 0,
                    "adults": 2,
                    "old_adults": 0,
                    "number": 4,
                },
                3: {
                    "kids": 1,
                    "young_adults": 0,
                    "adults": 2,
                    "old_adults": 1,
                    "number": 3,
                },
                4: {
                    "kids": 0,
                    "young_adults": 2,
                    "adults": 2,
                    "old_adults": 0,
                    "number": 2,
                },
            },
            "students": {"number": 10},
            "communal": {"number": 3},
            "other": {
                "kids": 0,
                "young_adults": "2+",
                "adults": "1+",
                "old_adults": 1,
                "number": 2,
            },
        }
        households = Households.from_household_compositions(compositions)
        return households

    def test__number_of_households(self, households):
        assert len(households) == 5 + 7 + 5 + 4 + 3 + 2 + 10 + 3 + 2

    def test__households_from_compositions(self, households):
        family_households = [
            household for household in households if household.type == "family"
        ]
        assert len(family_households) == 5 + 4 + 3 + 2
        kids = np.zeros(10)
        young_adults = np.zeros(10)
        adults = np.zeros(10)
        old_adults = np.zeros(10)
        for household in households:
            if household._kids == 0:
                kids[0] += 1
            elif household._kids == 1:
                kids[1] += 1
            else:
                kids[2] += 1
            if household._adults == 1:
                adults[1] += 1
            elif household._adults == 2:
                adults[2] += 1

            if household._old_adults == 0:
                old_adults[1] += 1

        assert kids[0] == 2
        assert kids[1] == 8
        assert kids[2] == 4

        assert adults[1] == 5
        assert adults[2] == 5

        assert old_adults[0] == 11
        assert old_adults[1] == 3



