from collections import OrderedDict, defaultdict
from itertools import chain, combinations
from typing import List, Union, Dict
from random import random, shuffle
import logging
import numpy as np
import numba as nb
import pandas as pd
import yaml
from scipy import stats

from june import paths
from june.exc import HouseholdError
from june.demography import Person
from june.demography.geography import Area
from june.groups.household import Household, Households, HouseholdComposition
from june.utils.numba_random import random_choice_numba

logger = logging.getLogger(__name__)

default_config_filename = (
    paths.configs_path / "defaults/distributors/household_distributor.yaml"
)

default_household_composition_filename = (
    paths.data_path / "input/households/household_composition_by_area.json"
)

default_number_students_filename = (
    paths.data_path / "input/households/number_students_per_area.csv"
)

default_couples_age_difference_filename = (
    paths.data_path / "input/households/couples_age_difference.csv"
)

default_parent_kid_age_difference_filename = (
    paths.data_path / "input/households/parent_kid_age_difference.csv"
)


def get_closest_element_and_distance_in_array(array, value):
    diff = np.abs(value - array)
    min_idx = np.argmin(diff)
    return array[min_idx], diff[min_idx]


@nb.njit()
def _sample_increase_in_resident_number_2(prob0=0.0, prob1=0.0):
    relative_prob = prob0 / (prob0 + prob1)
    if random() < relative_prob:
        return 0
    else:
        return 1


@nb.njit()
def _sample_increase_in_resident_number_1(prob0=0.0):
    if random() < prob0:
        return 0
    else:
        return -1


class PersonFinder:
    """
    Finds a person in the area with the given characteristics if that person
    has not been picked before (ie is not in a household)
    """

    def __init__(
        self,
        population: List[Person],
        first_kid_parent_age_difference: int = 25,
        second_kid_parent_age_difference: int = 30,
    ):
        self.population_dict = self._generate_population_dict(population)
        self.available_ethnicities = list("ABCDE")
        self.first_kid_parent_age_difference = first_kid_parent_age_difference
        self.second_kid_parent_age_difference = second_kid_parent_age_difference

    def find_partner(
        self,
        person: Person,
        match_ethnicity: bool = False,
        maximum_age_difference: int = 25,
    ):
        """
        Looks for a partner for the given person. If ``match_ethnicity`` is True, then return a person of the matching ethnicity, if possible.
        """
        if match_ethnicity and self.population_dict[person.ethnicity]:
            partner = self._find_partner_fixed_ethnicity(
                person, person.ethnicity, maximum_age_difference
            )
            if partner is not None:
                return partner
        # looking for partner with different ethnicity
        shuffle(self.available_ethnicities)
        for ethnicity in self.available_ethnicities:
            if ethnicity not in self.population_dict:
                continue
            partner = self._find_partner_fixed_ethnicity(
                person, ethnicity, maximum_age_difference
            )
            if partner is not None:
                return partner

    def _generate_population_dict(self, population):
        population_dict = {}
        for person in population:
            if person.ethnicity not in population_dict:
                population_dict[person.ethnicity] = {}
            if person.sex not in population_dict[person.ethnicity]:
                population_dict[person.ethnicity][person.sex] = {}
            if person.age not in population_dict[person.ethnicity][person.sex]:
                population_dict[person.ethnicity][person.sex][person.age] = []
            population_dict[person.ethnicity][person.sex][person.age].append(person)
        return population_dict

    def _pop_person(self, ethnicity, sex, age):
        """
        Pops a person and deletes the dictionary keys if there are no more
        people of that age / sex / ethnicity left.
        """
        person = self.population_dict[ethnicity][sex][age].pop()
        if not self.population_dict[ethnicity][sex][age]:
            del self.population_dict[ethnicity][sex][age]
            if not self.population_dict[ethnicity][sex]:
                del self.population_dict[ethnicity][sex]
                if not self.population_dict[ethnicity]:
                    del self.population_dict[ethnicity]
        return person

    def _opposite_sex(self, person):
        if person.sex == "m":
            return "f"
        else:
            return "m"

    def _find_partner_fixed_ethnicity(
        self, person: Person, ethnicity: str, maximum_age_difference: int = 25
    ):
        """
        Given a person and a target ethnicity, looks for a partner for the person thas has the given ethnicity.
        To determine the partner's age,
        Finds a partner for the given person the rules are:
        - Look for the opposite sex, if not available, then return the same sex.
        - Look for the closest age, if the difference is larger than ``maximum_age_difference``, 
          then look at the same sex.
          Always return people aged >= 18 years old.
        """
        population_age_sex = self.population_dict[ethnicity]
        if population_age_sex[self._opposite_sex(person)]:
            partner_sex = self._opposite_sex(person)
            if partner_sex in population_age_sex:
                available_ages = np.array(
                    [age for age in population_age_sex[partner_sex] if age >= 18]
                )
                partner_age, age_diff = get_closest_element_and_distance_in_array(
                    available_ages, person.age
                )
                if age_diff < maximum_age_difference:
                    return self._pop_person(ethnicity, partner_sex, partner_age)
        # looking for same sex partner
        partner_sex = person.sex
        if partner_sex in population_age_sex:
            available_ages = np.array(
                [age for age in population_age_sex[partner_sex] if age >= 18]
            )
            partner_age, age_diff = get_closest_element_and_distance_in_array(
                available_ages, person.age
            )
            if age_diff < maximum_age_difference:
                return self._pop_person(ethnicity, partner_sex, partner_age)


class HouseholdCompositionAdapter:
    """
    Class to store functions that adapt a census' household composition to a certain household
    size by sampling resident numbers from census data.
    """

    def __init__(self, probabilities_per_age_group):
        self.probabilities_per_age_group = probabilities_per_age_group

    @staticmethod
    def _parse_age_group_number(age_group: Union[int, tuple, str]) -> tuple:
        """
        Parses age group:
        4 -> (4,4)
        (2,3) -> (2,3)
        "2+" -> (2, inf)
        "3-5" -> (3,5)
        """
        if age_group is None:
            return (0, 0)
        if type(age_group) == str:
            if "+" in age_group:
                return (int(age_group[0]), np.inf)
            elif "-" in age_group:
                return (int(age_group[0]), int(age_group[2]))
            else:
                raise HouseholdError(
                    f"household composition coding {age_group} not supported."
                )
        elif type(age_group) == int:
            return (age_group, age_group)
        else:
            return age_group

    def _parse_composition_dict(self, composition_dict: dict) -> dict:
        ret = defaultdict(lambda: (0, 0))
        for key, value in composition_dict.items():
            ret[key] = self._parse_age_group_number(age_group=value)
        return ret

    def _get_composition_size(self, composition_dict: dict) -> int:
        return sum(age_group[0] for age_group in composition_dict.values())

    def adapt_household_composition_to_target_size(
        self, composition_dict: dict, target_size: int = None,
    ):

        """
        Adapts the current household composition dictionary to a target size.
        We do it by increasing the number of residents of the age groups that
        allow it, sampling from probabilities taken from census data, for example,
        the probability of having ``n`` number of kids.
    
        Parameters
        ----------
        probabilities_per_age_group
            a dictionary with the probability of having a certain number of residents
            of a particular age group
            Example:
            ```
            probabilities_per_age_group = {"kids" : {1 : 0.25, 2 : 0.5, 3 : 0.25}}
            ```
        target_size
            total number of people desired in the composition. If None, no constraint.
        """
        resident_ranges = self._parse_composition_dict(composition_dict)
        resident_numbers = defaultdict(lambda: 0)
        for age_group, resident_range in resident_ranges.items():
            resident_numbers[age_group] = resident_range[0]
        if target_size is None:
            for age_group in self.probabilities_per_age_group:
                n_residents_range = resident_ranges[age_group]
                if n_residents_range[1] > n_residents_range[0]:
                    resident_numbers[
                        age_group
                    ] = self._sample_increase_in_resident_number_for_age_group(
                        self.probabilities_per_age_group[age_group],
                        n_residents_range[0],
                    )
        else:
            resident_numbers = self._sample_increase_in_resident_number(
                resident_ranges=resident_ranges, target_size=target_size,
            )
        return resident_numbers

    def _sample_increase_in_resident_number_for_age_group(
        self,
        probabilities_per_resident_number: List[float],
        current_resident_number: int,
    ) -> int:
        """
        Given the current number of a certain resident age group, eg, kids, uses the
        probability of having a certain number of kids in a household to sample if
        we have a number increase or not, and returns it.
    
        Parameters
        ----------
        probabilities_per_resident_number
            probability for each resident number
        current_resident_number
            how many age group members we have now
        """
        if current_resident_number + 1 >= len(probabilities_per_resident_number):
            return current_resident_number
        probabilities_normed = probabilities_per_resident_number[
            current_resident_number:
        ] / np.sum(probabilities_per_resident_number[current_resident_number:])
        return random_choice_numba(
            np.arange(current_resident_number, len(probabilities_per_resident_number)),
            probabilities_normed,
        )

    def _sample_increase_in_resident_number(
        self, resident_ranges: Dict[str, tuple], target_size: int,
    ) -> Dict[str, int]:
        """
        Given the current number of residents per age group in a household composition,
        samples increases in the number of certain age groups until reaching the target size.

        Parameters
        ----------
        probabilities_per_resident_number
            probability for each resident number
        current_resident_number
            how many age group members we have now
        """
        resident_numbers = defaultdict(lambda: 0)
        for age_group, resident_range in resident_ranges.items():
            resident_numbers[age_group] = resident_range[0]
        current_size = sum(resident_numbers.values())
        while current_size < target_size:
            probabilities_of_extra_members = []
            age_groups = []
            for age_group, probabilities in self.probabilities_per_age_group.items():
                resident_range = resident_ranges[age_group]
                n_residents_age_group = resident_numbers[age_group]
                if n_residents_age_group >= resident_range[
                    1
                ] or n_residents_age_group + 1 > len(probabilities):
                    continue
                age_groups.append(age_group)
                probabilities_of_extra_member = np.sum(
                    probabilities[n_residents_age_group + 1 :]
                ) / np.sum(probabilities)
                probabilities_of_extra_members.append(probabilities_of_extra_member)
            if len(probabilities_of_extra_members) == 2:
                prob0, prob1 = probabilities_of_extra_members
                relative_prob = prob0 / (prob0 + prob1)
                if random() < relative_prob:
                    resident_numbers[age_groups[0]] += 1
                else:
                    resident_numbers[age_groups[1]] += 1
            elif len(probabilities_of_extra_members) == 1:
                resident_numbers[age_groups[0]] += 1
            else:
                break
            current_size += 1
        return resident_numbers


class HouseholdCompositionLinker:
    """
    Links household compositions to households of a given size.
    """

    def __init__(
        self,
        hc_adpater: HouseholdCompositionAdapter,
        order_of_linking: List[str] = None,
    ):
        self.hc_adpater = hc_adpater
        if order_of_linking is None:
            order_of_linking = ["single", "couple", "family", "student", "shared"]

    def _get_household_from_size_dict(self, households_per_size, size):
        hsize = None
        for available_size in households_per_size:
            if available_size >= size:
                hsize = available_size
                break
        if hsize is None:
            raise HouseholdError(
                "Cannot find a household for given composition!"
                f"Available household sizes {households_per_size.keys()}, asked size: {size}"
            )
        household = households_per_size[hsize].pop()
        if not households_per_size[hsize]:
            del households_per_size[hsize]
        return household

    def link_family_compositions(self, composition_dict: dict, households: Households):
        """
        Links family household compositions with a ceratin household.
        If the composition is certain, ie, the number of residents per age group is known exactly,
        then the composition is link to household of the proper size.
        Otherwise, we link to the household that has the closest superior size, and adapt the
        composition to the actual household size using the HouseholdCompositionAdapter
        """
        households_per_size = defaultdict(list)
        for household in households:
            if household.composition is None:
                households_per_size[household.max_size].append(household)
        uncertain_compositions = []
        for _, composition in composition_dict.items():
            composition_number = composition.pop("number")
            for _ in range(composition_number):
                fixed_size = True
                for value in composition.values():
                    if type(value) == str:
                        fixed_size = False
                        break
                if fixed_size:
                    hsize = sum(composition.values())
                    household = self._get_household_from_size_dict(
                        households_per_size=households_per_size, size=hsize
                    )
                    household.composition = HouseholdComposition.from_dict(
                        composition, household_type="family"
                    )
                else:
                    uncertain_compositions.append(composition)
        for composition in uncertain_compositions:
            minimum_size = sum(
                range[0]
                for range in self.hc_adpater._parse_composition_dict(
                    composition
                ).values()
            )
            household = self._get_household_from_size_dict(
                households_per_size=households_per_size, size=minimum_size
            )
            adapted_comp = self.hc_adpater.adapt_household_composition_to_target_size(
                composition_dict=composition, target_size=household.max_size
            )
            household.composition = HouseholdComposition.from_dict(
                adapted_comp, household_type="family"
            )

    def link_single_compositions(self, composition_dict: dict, households: Households):
        households_per_size = defaultdict(list)
        for household in households:
            if household.composition is None:
                households_per_size[household.max_size].append(household)
        for _, composition in composition_dict.items():
            composition_number = composition.pop("number")
            for _ in range(composition_number):
                hsize = sum(composition.values())
                household = self._get_household_from_size_dict(
                    households_per_size=households_per_size, size=hsize
                )
                household.composition = HouseholdComposition.from_dict(
                    composition, household_type="single"
                )

    def link_single_compositions(self, composition_dict: dict, households: Households):
        households_per_size = defaultdict(list)
        for household in households:
            if household.composition is None:
                households_per_size[household.max_size].append(household)
        for _, composition in composition_dict.items():
            composition_number = composition.pop("number")
            for _ in range(composition_number):
                hsize = sum(composition.values())
                household = self._get_household_from_size_dict(
                    households_per_size=households_per_size, size=hsize
                )
                household.composition = HouseholdComposition.from_dict(
                    composition, household_type="single"
                )

    def link_couple_compositions(self, composition_dict: dict, households: Households):
        households_per_size = defaultdict(list)
        for household in households:
            if household.composition is None:
                households_per_size[household.max_size].append(household)
        for _, composition in composition_dict.items():
            composition_number = composition.pop("number")
            for _ in range(composition_number):
                hsize = sum(composition.values())
                household = self._get_household_from_size_dict(
                    households_per_size=households_per_size, size=hsize
                )
                household.composition = HouseholdComposition.from_dict(
                    composition, household_type="couple"
                )

    def link_student_compositions(self, composition_dict: dict, households: Households):
        available_houses = [
            household for household in households if household.composition is None
        ]
        available_sizes = [household.max_size for household in available_houses]
        n_students = composition_dict["residents"]
        n_households = composition_dict["number"]
        # need to pick n_households that sum to n_students!
        # this part might not be very efficient...
        # brute force!
        found_combination = False
        for i in range(1, n_households+1):
            # trying combination of i households
            size_combinations = combinations(available_sizes, i)
            for combination in size_combinations:
                if sum(combination) == n_students:
                    found_combination = True
                    break
            if found_combination:
                break
        if not found_combination:
            raise HouseholdError(
                f"Can't find student household combination to fit people in."
            )
        combination = list(combination)
        for household in available_houses:
            if household.max_size in combination:
                combination.remove(household.max_size)
                household.composition = HouseholdComposition(
                    young_adults=household.max_size, household_type="student"
                )


class HouseholdDistributor:
    pass
