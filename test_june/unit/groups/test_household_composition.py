import pytest
import numpy as np
from collections import defaultdict

from june.groups.household import HouseholdComposition, HC


@pytest.fixture()
def make_full_dict():
    compositions = {
        "single": {1: {"old_adults": 1, "number": 10}, 2: {"adults": 1, "number": 5}},
        "couple": {
            1: {"old_adults": 2, "number": 15},
            2: {"adults": 2, "number": 7},
        },
        "family": {
            0: {"kids": 1, "adults": 2, "number": 5,},
            1: {"kids": "2+", "adults": 2, "number": 4,},
            2: {"kids": 1, "adults": 2, "old_adults": 1, "number": 3,},
            3: {
                "kids": 0,
                "young_adults": 2,
                "adults": 2,
                "old_adults": 0,
                "number": 2,
            },
            4: {"kids": "1+", "young_adults": "0+", "adults": "1+", "old_adults": "1+"},
            5: {"kids": "2+", "young_adults": "0+", "adults": "1+", "old_adults": "1+"},
        },
        "student": {"residents": 40, "number": 10},
        "communal": {"residents": 10, "number": 3},
        "shared": {"all_adults": "2+", "number": 2,},
    }


@pytest.fixture(name="simple_composition")
def make_simple_dict():
    compositions = {
        "single": {1: {"old_adults": 1, "number": 10}},
        "couple": {1: {"old_adults": 2, "number": 15},},
        "family": {0: {"kids": 1, "adults": 2, "number": 5,},},
        "student": {"residents": 40, "number": 10},
        "communal": {"residents": 10, "number": 3},
        "shared": {"all_adults": "2+", "number": 2,},
    }


class TestHouseholdComposition:
    def test__initialization(self):
        hc = HouseholdComposition(young_adults=0, adults=2, household_type="couple")
        assert hc.size == 2
        assert hc.kids == 0
        assert hc.young_adults == 0
        assert hc.adults == 2
        assert hc.old_adults == 0
        assert hc.household_type == HC.couple

        hc = HouseholdComposition(
            kids=1,
            young_adults=4,
            adults=3,
            old_adults=3,
            household_type="family",
        )
        assert hc.size == 11
        assert hc.kids == 1
        assert hc.young_adults == 4
        assert hc.adults == 3
        assert hc.old_adults == 3
        assert hc.household_type == HC.family

    def test__from_dict(self):
        hc = HouseholdComposition.from_dict(
            composition_dict={
                "kids": 2,
                "young_adults": 0,
                "adults": 2,
                "old_adults": 1,
            },
            household_type="family",
        )
        assert hc.household_type == HC.family
        assert hc.kids == 2
        assert hc.young_adults == 0
        assert hc.adults == 2
        assert hc.old_adults == 1
        assert hc.size == 5

