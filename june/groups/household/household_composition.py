from typing import Union, List, Dict
from enum import IntEnum
from random import random
import numpy as np
import numba as nb

from june.exc import HouseholdError


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
        kids: int = None,
        young_adults: int  = None,
        adults: int = None,
        old_adults: int  = None,
    ):
        # if type is not None:
        if household_type is not None:
            self.household_type = getattr(HC, household_type)
        self.kids = kids or 0
        self.young_adults = young_adults or 0
        self.adults = adults or 0
        self.old_adults = old_adults or 0


    @property
    def size(self) -> int:
        return self.kids + self.young_adults + self.adults + self.old_adults

    @classmethod
    def from_dict(
        cls, composition_dict: dict, household_type: str
    ) -> "HouseholdComposition":
        """
        Parses household composition from dict. An example of a compatible dict is:
        HouseholdComposition.from_dict(
            composition_dict={
                "kids": 2,
                "young_adults": 0,
                "adults": 2,
                "old_adults": 1,
            },
            household_type="family",
        )
        """
        return cls(household_type=household_type, **composition_dict)

