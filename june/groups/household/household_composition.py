from typing import Union, List, Dict
from enum import IntEnum
from random import random
import numpy as np
import numba as nb

from june.exc import HouseholdError


@nb.njit()
def sample_increase_in_resident_number(prob0=0.0, prob1=0.0):
    relative_prob = prob0 / (prob0 + prob1)
    if random() < relative_prob:
        return 0
    else:
        return 1

@nb.njit()
def sample_increase_in_resident_number(prob0=0.0):
    if random() < prob0:
        return 0
    else:
        return -1


class HC(IntEnum):
    """
    Defines the household type
    """

    single = 0
    couple = 1
    family = 2
    shared = 3
    student = 4
    communal = 5


class HouseholdComposition:
    """
    This class represents a household composition.
    A household composition is defined by a type
    and a number of residents for each age group.

    Parameters
    ----------
    household_type:
       household type. Available are: 
        - single
        - couple
        - family
        - shared
        - student
        - communal
    kids:
        an int or tuple defining the range number of kids
    young_adults:
        an int or tuple defining the range number of young adults
    adults:
        an int or tuple defining the range number of adults
    old_adults:
        an int or tuple defining the range number of old adults
    """

    def __init__(
        self,
        household_type: str = None,
        kids: Union[int, tuple] = None,
        young_adults: Union[int, tuple] = None,
        adults: Union[int, tuple] = None,
        old_adults: Union[int, tuple] = None,
    ):
        # if type is not None:
        if household_type is not None:
            self.household_type = getattr(HC, household_type)
        self.kids = self._parse_age_group_number(kids)
        self.young_adults = self._parse_age_group_number(young_adults)
        self.adults = self._parse_age_group_number(adults)
        self.old_adults = self._parse_age_group_number(old_adults)

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

    @staticmethod
    def _parse_all_age_group_numbers(composition_dict: dict) -> dict:
        ret = {}
        for key, value in composition_dict.items():
            ret[key] = HouseholdComposition._parse_age_group_number(value)
        return ret

    @property
    def min_size(self) -> int:
        return self.kids[0] + self.young_adults[0] + self.adults[0] + self.old_adults[0]

    @property
    def max_size(self) -> int:
        return self.kids[1] + self.young_adults[1] + self.adults[1] + self.old_adults[1]

    @classmethod
    def from_dict(
        cls, composition_dict: dict, household_type: str
    ) -> "HouseholdComposition":
        """
        Parses household composition from dict. The number ranges can be specified
        using encodings like 4+ and 2-3. An example of a compatible dict is:
        HouseholdComposition.from_dict(
            composition_dict={
                "kids": "0-2",
                "young_adults": "0+",
                "adults": 2,
                "old_adults": "0-1",
            },
            household_type="family",
        )
        """
        return cls(household_type=household_type, **composition_dict)

    def adapt_to_target_size(
        self,
        probabilities_per_age_group: dict,
        target_size=None,
    ) -> None:
        """
        Adapts the current household composition to the target size.
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
        current_resident_number = self.min_size
        if target_size is None:
            if self.kids[1] > self.kids[0]:
                n_kids = self._sample_increase_in_resident_number_for_age_group(
                    probabilities_per_age_group["kids"], self.kids[0]
                )
            if self.young_adults[1] > self.young_adults[0]:
                n_young_adults = self._sample_increase_in_resident_number_for_age_group(
                    probabilities_per_age_group["young_adults"], self.young_adults[0]
                )
        else:
            resident_numbers = self._sample_increase_in_resident_number(
                current_resident_number=current_resident_number,
                probabilities_per_age_group=probabilities_per_age_group,
                target_size=target_size,
            )
            for key, value in resident_numbers.items():
                setattr(self, key, value)

    def _sample_increase_in_resident_number_for_age_group(
        self,
        probabilities_per_resident_number: Dict[int, float],
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
        while True:
            probability_of_extra_member = probabilities_per_resident_number.get(
                current_resident_number + 1, 0
            )
            if random() < probability_of_extra_member:
                current_resident_number += 1
            else:
                break
        return current_resident_number

    def _sample_increase_in_resident_number(
        self,
        probabilities_per_age_group: Dict[str, Dict[int, float]],
        current_resident_number: Dict[str, int],
        target_size: int,
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
        current_size = sum(
            age_group_range[0] for age_group_range in current_resident_number.values()
        )
        while current_size < target_size:
            probabilities_of_extra_members = []
            age_groups = []
            for age_group in probabilities_per_age_group:
                if getattr(self, age_group)[0] >= getattr(self, age_group)[1]:
                    continue
                age_groups.append(age_group)
                probability_of_extra_member = probabilities_per_age_group.get(
                    current_resident_number[age_group] + 1, 0
                )
                probabilities_of_extra_members.append(probabilities_of_extra_members)
            if probabilities_of_extra_members:
                ret = sample_increase_in_resident_number(
                    *probabilities_of_extra_members
                )
                if ret == -1:
                    break
            current_resident_number[age_groups[ret]] += 1
            current_size += 1
        return current_resident_number
