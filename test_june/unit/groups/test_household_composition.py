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
            2: {"old_adults": "0+", "adults": "1+", "number": 7},
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
        assert hc.min_size == 2
        assert hc.max_size == 2
        assert hc.kids == (0, 0)
        assert hc.young_adults == (0, 0)
        assert hc.adults == (2, 2)
        assert hc.old_adults == (0, 0)
        assert hc.household_type == HC.couple

        hc = HouseholdComposition(
            kids=(0, 1), young_adults=(1, 4), adults=(2, 3), old_adults=3, household_type="family"
        )
        assert hc.min_size == 6
        assert hc.max_size == 11
        assert hc.kids == (0, 1)
        assert hc.young_adults == (1, 4)
        assert hc.adults == (2, 3)
        assert hc.old_adults == (3, 3)
        assert hc.household_type == HC.family

    def test__from_dict(self):
        hc = HouseholdComposition.from_dict(
            composition_dict={
                "kids": "0-2",
                "young_adults": "0+",
                "adults": 2,
                "old_adults": "0-1",
            },
            household_type="family",
        )
        assert hc.household_type == HC.family
        assert hc.kids == (0, 2)
        assert hc.young_adults == (0, np.inf)
        assert hc.adults == (2, 2)
        assert hc.old_adults == (0, 1)
        assert hc.min_size == 2
        assert hc.max_size == np.inf

    def test__compute_actual_household_composition_from_demographics(self):
        probabilities_per_kid = {1: 0.6, 2: 0.4}
        probabilities_per_young_adult = {0: 0.5, 1: 0.5}
        composition = {"kids": "2+", "young_adults": "0+", "adults": 2, "old_adults": 0}
        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        for _ in range(100):
            real_composition = HouseholdComposition.from_demographics(
                composition_dict=composition,
                probabilities_per_kid=probabilities_per_kid,
                probabilities_per_young_adult=probabilities_per_young_adult,
                target_size = None
            )
            n_kids[real_composition["kids"]] += 1
            n_young_adults[real_composition["young_adults"]] += 1
            n_adults[real_composition["adults"]] += 1
            n_old_adults[real_composition["old_adults"]] += 1
            assert real_composition.size >= 4
        assert len(n_adults) == 1
        assert n_adults[2] == 100
        assert len(n_old_adults) == 1
        assert n_old_adults[2] == 100
        assert np.isclose(n_kids[2], 60, atol=5)
        assert np.isclose(n_kids[3], 40, atol=5)
        assert np.isclose(n_young_adults[0], 50, atol=5)
        assert np.isclose(n_young_adults[1], 50, atol=5)

    def test__compute_actual_household_composition_from_demographics_with_size_constraint(self):
        probabilities_per_kid = {1: 0.6, 2: 0.4}
        probabilities_per_young_adult = {0: 0.5, 1: 0.5}
        composition = {"kids": "2+", "young_adults": "0+", "adults": 2, "old_adults": 0}
        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        for _ in range(100):
            real_composition = HouseholdComposition.from_demographics(
                composition_dict=composition,
                household_type = "family",
                probabilities_per_kid=probabilities_per_kid,
                probabilities_per_young_adult=probabilities_per_young_adult,
                target_size = 4
            )
            n_kids[real_composition["kids"]] += 1
            n_young_adults[real_composition["young_adults"]] += 1
            n_adults[real_composition["adults"]] += 1
            n_old_adults[real_composition["old_adults"]] += 1
            assert real_composition.size == 4
        assert len(n_adults) == 1
        assert n_adults[2] == 100
        assert len(n_old_adults) == 1
        assert n_old_adults[2] == 100
        assert len(n_kids) == 1
        assert n_kids[2] == 100
        assert len(n_young_adults) == 1
        assert n_young_adults[0] == 100

        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        for _ in range(100):
            real_composition = HouseholdComposition.from_demographics(
                composition_dict=composition,
                household_type = "family",
                probabilities_per_kid=probabilities_per_kid,
                probabilities_per_young_adult=probabilities_per_young_adult,
                target_size = 5
            )
            if real_composition["kids"] == 3:
                assert real_composition["young_adults"] == 0
            if real_composition["young_adults"] == 1:
                assert real_composition["kids"] == 2
            assert real_composition.size == 5
            n_kids[real_composition["kids"]] += 1
            n_young_adults[real_composition["young_adults"]] += 1
            n_adults[real_composition["adults"]] += 1
            n_old_adults[real_composition["old_adults"]] += 1
        assert len(n_adults) == 1
        assert n_adults[2] == 100
        assert len(n_old_adults) == 1
        assert n_old_adults[2] == 100
        assert len(n_kids) == 1
        assert n_kids[2] == 100
        assert len(n_young_adults) == 1
        assert n_young_adults[0] == 100



class TestHouseholdCompositionPairing:
    def test__composition_paring(self):
        pass
