import os
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pytest

from june.demography.person import Person
from june.demography import Demography
from june.demography.geography import Geography
from june.groups import Household, Households
from june.groups.household import HouseholdComposition
from june.distributors import (
    HouseholdDistributor,
    PersonFinder,
    HouseholdCompositionAdapter,
    HouseholdCompositionLinker
)

@pytest.fixture(name="hc_adapter")
def make_adapter():
    probabilities_per_kid = [0.1, 0.2, 0.3, 0.4]
    probabilities_per_young_adult = [0.2, 0.4, 0.3, 0.1]
    probabilities_per_age_group = {
        "kids": probabilities_per_kid,
        "young_adults": probabilities_per_young_adult,
    }
    return HouseholdCompositionAdapter(
        probabilities_per_age_group=probabilities_per_age_group
    )

@pytest.fixture(name="hc_linker")
def make_hl(hc_adapter):
    return HouseholdCompositionLinker(hc_adapter)

class TestPersonFinder:
    def test__find_partner(self):
        person = Person(sex="m", age=18, ethnicity="A")
        population = [
            Person.from_attributes(sex="f", age=17, ethnicity="A"),
            Person.from_attributes(sex="f", age=19, ethnicity="B"),
            Person.from_attributes(sex="f", age=21, ethnicity="A"),
            Person.from_attributes(sex="f", age=50, ethnicity="A"),
            Person.from_attributes(sex="f", age=50, ethnicity="C"),
        ]
        person_finder = PersonFinder(population)
        p = person_finder.find_partner(person, match_ethnicity=False)
        assert p.age == 19
        assert p.sex == "f"
        person_finder = PersonFinder(population)
        p = person_finder.find_partner(person, match_ethnicity=True)
        assert p.age == 21
        assert p.sex == "f"
        assert p.ethnicity == "A"


class TestHouseholdCompositionsFromCensus:

    def test__compute_actual_household_composition_from_demographics(self, hc_adapter):
        composition = {"kids": "0+", "young_adults": "0+", "adults": 2, "old_adults": 0}
        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        n_samples = 1000
        for _ in range(n_samples):
            adapted_comp = hc_adapter.adapt_household_composition_to_target_size(
                composition_dict=composition,
                target_size=None,
            )
            n_kids[adapted_comp["kids"]] += 1
            n_young_adults[adapted_comp["young_adults"]] += 1
            n_adults[adapted_comp["adults"]] += 1
            n_old_adults[adapted_comp["old_adults"]] += 1
        assert len(n_adults) == 1
        assert n_adults[2] == n_samples
        assert len(n_old_adults) == 1
        assert n_old_adults[0] == n_samples
        rtol = 0.15
        assert np.isclose(n_kids[0], 0.1 * n_samples, rtol=rtol)
        assert np.isclose(n_kids[1], 0.2 * n_samples, rtol=rtol)
        assert np.isclose(n_kids[2], 0.3 * n_samples, rtol=rtol)
        assert np.isclose(n_kids[3], 0.4 * n_samples, rtol=rtol)
        assert np.isclose(n_young_adults[0], 0.2 * n_samples, rtol=rtol)
        assert np.isclose(n_young_adults[1], 0.4 * n_samples, rtol=rtol)
        assert np.isclose(n_young_adults[2], 0.3 * n_samples, rtol=rtol)
        assert np.isclose(n_young_adults[3], 0.1 * n_samples, rtol=rtol)

    def test__compute_actual_household_composition_from_demographics_with_size_constraint(
        self, hc_adapter
    ):
        composition = {"kids": "2+", "young_adults": "0+", "adults": 2, "old_adults": 0}
        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        n_samples = 1000
        for _ in range(n_samples):
            adapted_comp = hc_adapter.adapt_household_composition_to_target_size(
                composition_dict=composition,
                target_size=4,
            )
            n_kids[adapted_comp["kids"]] += 1
            n_young_adults[adapted_comp["young_adults"]] += 1
            n_adults[adapted_comp["adults"]] += 1
            n_old_adults[adapted_comp["old_adults"]] += 1
            assert sum(adapted_comp.values()) == 4
        assert len(n_adults) == 1
        assert n_adults[2] == n_samples
        assert len(n_old_adults) == 1
        assert n_old_adults[0] == n_samples
        assert len(n_kids) == 1
        assert n_kids[2] == n_samples
        assert len(n_young_adults) == 1
        assert n_young_adults[0] == n_samples

        composition = {"kids": "1+", "young_adults": "1+", "adults": 2, "old_adults": 0}
        n_kids = defaultdict(int)
        n_young_adults = defaultdict(int)
        n_adults = defaultdict(int)
        n_old_adults = defaultdict(int)
        n_samples = 1000
        for _ in range(n_samples):
            adapted_comp = hc_adapter.adapt_household_composition_to_target_size(
                composition_dict=composition,
                target_size=5,
            )
            n_kids[adapted_comp["kids"]] += 1
            n_young_adults[adapted_comp["young_adults"]] += 1
            n_adults[adapted_comp["adults"]] += 1
            n_old_adults[adapted_comp["old_adults"]] += 1
            assert sum(adapted_comp.values()) == 5
        assert len(n_adults) == 1
        assert n_adults[2] == n_samples
        assert len(n_old_adults) == 1
        assert n_old_adults[0] == n_samples
        assert len(n_kids) == 2
        assert n_kids[1] > 0
        assert n_kids[2] > 0
        assert len(n_young_adults) == 2
        assert n_young_adults[1] > 0
        assert n_young_adults[2] > 0
        assert np.isclose(n_kids[2], 2 * n_young_adults[2], rtol=0.1)

class TestLinkHouseholdsAndCompositions:
    def test__link_family_households(self, hc_linker):
        households = Households([])
        for size, number in zip([3, 5], [5,4]):
            for _ in range(number):
                households.add(Household(size=size, type="family"))
        composition_dict = {"family": {
            0: {"kids": 1, "adults": 2, "number": 5,},
            1: {"kids": "2+", "adults": 2, "number": 4},
            #2: {"kids": 1, "adults": 2, "old_adults": 1, "number": 3,},
            #3: {
            #    "kids": 0,
            #    "young_adults": 2,
            #    "adults": 2,
            #    "old_adults": 0,
            #    "number": 2,
            #},
            #4: {"kids": "1+", "young_adults": "0+", "adults": "1+", "old_adults": "1+", "number":1},
            #5: {"kids": "2+", "young_adults": "0+", "adults": "1+", "old_adults": "1+", "number":1},
        }}
        hc_linker.link_family_compositions(composition_dict["family"], households)
        for household in households:
            if household.max_size == 3:
                assert household.composition.kids == 1
                assert household.composition.adults == 2
                assert household.composition.young_adults == 0
                assert household.composition.old_adults == 0
            else:
                assert household.max_size == 5
                assert household.composition.kids == 3
                assert household.composition.adults == 2
                assert household.composition.young_adults == 0
                assert household.composition.old_adults == 0


