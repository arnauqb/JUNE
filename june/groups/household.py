import pandas as pd
import numpy as np
from enum import IntEnum
from typing import List, Union

from june import paths
from june.groups import Group, Supergroup
from june.demography import Person, Geography
from june.demography.geography import Area

default_household_sizes_filename = paths.data_path / "input/households/household_size_per_area.csv"

class Household(Group):
    __slots__ = (
        "type",
        "area",
        "residents",
    )

    class SubgroupType(IntEnum):
        kids = 0
        young_adults = 1
        adults = 2
        old_adults = 3

    def __init__(
        self,
        area: Area = None,
        size: int = None,
    ):
        super().__init__()
        if size is None:
            self.max_size = np.inf
        else:
            self.max_size = size
        self.area = area
        self.residents = ()

    def add(self, person, activity="residence"):
        if person.age <= self.kid_max_age:
            subgroup = self.SubgroupType.kids
        elif person.age <= self.young_adult_max_age:
            subgroup = self.SubgroupType.young_adults
        elif person.age <= self.adult_max_age:
            subgroup = self.SubgroupType.adults
        else:
            subgroup = self.SubgroupType.old_adults
        self[subgroup].append(person)
        if activity == "leisure":
            person.subgroups.leisure = self[subgroup]
        elif activity == "residence":
            person.subgroups.residence = self[subgroup]
            self.residents = tuple((*self.residents, person))
        else:
            raise ValueError("activity not supported")

    def get_leisure_subgroup(self, person):
        if person.age <= self.kid_max_age:
            return self.subgroups[self.SubgroupType.kids]
        elif person.age <= self.young_adult_max_age:
            return self.subgroups[self.SubgroupType.young_adults]
        elif person.age <= self.adult_max_age:
            return self.subgroups[self.SubgroupType.adults]
        else:
            return self.subgroups[self.SubgroupType.old_adults]

    @property
    def max_size(self):
        return self.max_size

    @property
    def kids(self):
        return self.subgroups[self.SubgroupType.kids]

    @property
    def n_kids(self):
        return len(self.kids)

    @property
    def young_adults(self):
        return self.subgroups[self.SubgroupType.young_adults]

    @property
    def n_young_adults(self):
        return len(self.young_adults)

    @property
    def adults(self):
        return self.subgroups[self.SubgroupType.adults]

    @property
    def n_adults(self):
        return len(self.adults)

    @property
    def old_adults(self):
        return self.subgroups[self.SubgroupType.old_adults]

    @property
    def n_old_adults(self):
        return len(self.old_adults)

class Households(Supergroup):
    def __init__(self, households: List[Household]):
        super().__init__()
        self.members = households

    @classmethod
    def _create_household(area, size):
        household = Household(area=area, size=int(size))
        area.households.append(household)
        return household

    @classmethod
    def for_geography(cls, geography: Geography, household_size_per_area_filename: str = default_household_sizes_filename):
        household_size_per_area = pd.read_csv(household_size_per_area_filename, index_col=0)
        for area in geography.areas:
            households = []
            sizes = household_size_per_area.loc[area.name]
            for size, size_number in sizes.items():
                for _ in range(int(size_number)):
                    household = cls._create_household(area=area, size=size)
                    households.append(household)
        return cls(households)


#def _str2class(str):
#    return getattr(sys.modules[__name__], str)
#
#
#def _parse_household_config_value(value: Union[int, str]):
#    if type(value) == str:
#        if "+" in value:
#            return (int(value.split("+")[0]), np.inf)
#    return value
#
#
#def _parse_composition_config(composition: dict):
#    ret = {}
#    for key in composition:
#        if key in ["n_kids", "n_young_adults", "n_adults", "n_old_adults"]:
#            ret[key + "_range"] = _parse_household_config_value(composition[key])
#    return ret


#class HouseholdComposition:
#    """
#    This class represents the household composition we would expect from census data.
#    It might not actually be the final household composition of a household, since we may 
#    not be able to match the population perfectly.
#
#    Parameters
#    ----------
#    n_kids_range: 
#        if int, then the allowed number of kids in the house, if tuple, the number range allowed.
#    n_young_adults_range: 
#        if int, then the allowed number of young adults in the house, if tuple, the number range allowed.
#    n_adults_range: 
#        if int, then the allowed number of adults in the house, if tuple, the number range allowed.
#    n_adults_range: 
#        if int, then the allowed number of old adults in the house, if tuple, the number range allowed.
#    """
#
#    def __init__(
#        self,
#        n_kids_range: Union[int, tuple] = None,
#        n_young_adults_range: Union[int, tuple] = None,
#        n_adults_range: Union[int, tuple] = None,
#        n_old_adults_range: Union[int, tuple] = None,
#    ):
#        if type(n_kids_range) == int:
#            self.n_kids_range = (n_kids_range, n_kids_range)
#        else:
#            self.n_kids_range = n_kids_range
#
#        if type(n_young_adults_range) == int:
#            self.n_young_adults_range = (n_young_adults_range, n_young_adults_range)
#        else:
#            self.n_young_adults_range = n_young_adults_range
#
#        if type(n_adults_range) == int:
#            self.n_adults_range = (n_adults_range, n_adults_range)
#        else:
#            self.n_adults_range = n_adults_range
#
#        if type(n_old_adults_range) == int:
#            self.n_old_adults_range = (n_old_adults_range, n_old_adults_range)
#        else:
#            self.n_old_adults_range = n_old_adults_range
#
#    @property
#    def max_size(self):
#        attr_names = [
#            "n_kids_range",
#            "n_young_adults_range",
#            "n_adults_range",
#            "n_old_adults_range",
#        ]
#        return sum(
#            [
#                getattr(self, name)[1]
#                for name in attr_names
#                if getattr(self, name) is not None
#            ]
#        )
#
#
#class Household(Group):
#    __slots__ = (
#        "type",
#        "area",
#        "kid_max_age",
#        "young_adult_max_age",
#        "adult_max_age",
#        "residents",
#        "composition",
#    )
#
#    class SubgroupType(IntEnum):
#        kids = 0
#        young_adults = 1
#        adults = 2
#        old_adults = 3
#
#    def __init__(
#        self,
#        area: Area = None,
#        n_kids_range: Union[int, tuple] = None,
#        n_young_adults_range: Union[int, tuple] = None,
#        n_adults_range: Union[int, tuple] = None,
#        n_old_adults_range: Union[int, tuple] = None,
#    ):
#        super().__init__()
#        self.spec = "household"
#        # composition according to census:
#        self.composition = HouseholdComposition(
#            n_kids_range=n_kids_range,
#            n_young_adults_range=n_young_adults_range,
#            n_adults_range=n_adults_range,
#            n_old_adults_range=n_old_adults_range,
#        )
#        self.type = self.get_spec().split("_")[-1]
#        self.area = area
#        self.kid_max_age = 17
#        self.young_adult_max_age = 34
#        self.adult_max_age = 64
#        self.residents = ()
#
#    def add(self, person, activity="residence"):
#        if person.age <= self.kid_max_age:
#            subgroup = self.SubgroupType.kids
#        elif person.age <= self.young_adult_max_age:
#            subgroup = self.SubgroupType.young_adults
#        elif person.age <= self.adult_max_age:
#            subgroup = self.SubgroupType.adults
#        else:
#            subgroup = self.SubgroupType.old_adults
#        self[subgroup].append(person)
#        if activity == "leisure":
#            person.subgroups.leisure = self[subgroup]
#        elif activity == "residence":
#            person.subgroups.residence = self[subgroup]
#            self.residents = tuple((*self.residents, person))
#        else:
#            raise ValueError("activity not supported")
#
#    def get_leisure_subgroup(self, person):
#        if person.age <= self.kid_max_age:
#            return self.subgroups[self.SubgroupType.kids]
#        elif person.age <= self.young_adult_max_age:
#            return self.subgroups[self.SubgroupType.young_adults]
#        elif person.age <= self.adult_max_age:
#            return self.subgroups[self.SubgroupType.adults]
#        else:
#            return self.subgroups[self.SubgroupType.old_adults]
#
#    @property
#    def max_size(self):
#        return self.composition.max_size
#
#    @property
#    def kids(self):
#        return self.subgroups[self.SubgroupType.kids]
#
#    @property
#    def n_kids(self):
#        return len(self.kids)
#
#    @property
#    def young_adults(self):
#        return self.subgroups[self.SubgroupType.young_adults]
#
#    @property
#    def n_young_adults(self):
#        return len(self.young_adults)
#
#    @property
#    def adults(self):
#        return self.subgroups[self.SubgroupType.adults]
#
#    @property
#    def n_adults(self):
#        return len(self.adults)
#
#    @property
#    def old_adults(self):
#        return self.subgroups[self.SubgroupType.old_adults]
#
#    @property
#    def n_old_adults(self):
#        return len(self.old_adults)
#
#
#class HouseholdSingle(Household):
#    def __init__(self, area: Area = None, old: bool = False):
#        if old:
#            super().__init__(
#                area=area,
#                n_kids_range=0,
#                n_young_adults_range=0,
#                n_adults_range=0,
#                n_old_adults_range=1,
#            )
#        else:
#            super().__init__(
#                area=area,
#                n_kids_range=0,
#                n_young_adults_range=(0, 1),
#                n_adults_range=(0, 1),
#                n_old_adults_range=0,
#            )
#
#    @property
#    def max_size(self):
#        return 1
#
#
#class HouseholdCouple(Household):
#    def __init__(self, area: Area = None, old=False):
#        if old:
#            super().__init__(
#                area=area,
#                n_kids_range=0,
#                n_young_adults_range=0,
#                n_adults_range=0,
#                n_old_adults_range=2,
#            )
#        else:
#            super().__init__(
#                area=area,
#                n_kids_range=0,
#                n_young_adults_range=(1, 2),
#                n_adults_range=(1, 2),
#                n_old_adults_range=(0, 1),
#            )
#
#    @property
#    def max_size(self):
#        return 2
#
#
#class HouseholdFamily(Household):
#    def __init__(
#        self,
#        area: Area = None,
#        n_kids_range=(1, 1),
#        n_parents=2,
#        n_young_adults_range=(0, 0),
#        n_old_adults_range=(0, 0),
#    ):
#        if (
#            type(n_young_adults_range) == tuple and n_young_adults_range[0] > 0
#            or type(n_young_adults_range) == int and n_young_adults_range > 0
#        ):  # non-dependent kids, require older parents
#            super().__init__(
#                area=area,
#                n_kids_range=n_kids_range,
#                n_young_adults_range=n_young_adults_range,
#                n_adults_range=(n_parents, n_parents),
#                n_old_adults_range=n_old_adults_range,
#            )
#        else:
#            super().__init__(
#                area=area,
#                n_kids_range=n_kids_range,
#                n_young_adults_range=(0, n_parents),
#                n_adults_range=(0, n_parents),
#                n_old_adults_range=n_old_adults_range,
#            )
#
#
#class HouseholdStudent(Household):
#    def __init__(self, area=None, n_students=0):
#        super().__init__(
#            area=area,
#            n_kids_range=0,
#            n_young_adults_range=(n_students, n_students),
#            n_adults_range=0,
#            n_old_adults_range=0,
#        )
#
#
#class HouseholdCommunal(Household):
#    def __init__(
#        area=None,
#        n_kids_range=(0, np.inf),
#        n_young_adults_range=(0, np.inf),
#        n_adults_range=(0, np.inf),
#        n_old_adults_range=(0, np.inf),
#    ):
#        super().__init__(
#            area=area,
#            n_kids_range=n_kids_range,
#            n_young_adults_range=n_young_adults_range,
#            n_adults_range=n_adults_range,
#            n_old_adults_range=n_old_adults_range,
#        )
#
#
#class HouseholdOther(Household):
#    def __init__(
#        area=None,
#        n_kids_range=(0, np.inf),
#        n_young_adults_range=(0, np.inf),
#        n_adults_range=(0, np.inf),
#        n_old_adults_range=(0, np.inf),
#    ):
#        super().__init__(
#            area=area,
#            n_kids_range=n_kids_range,
#            n_young_adults_range=n_young_adults_range,
#            n_adults_range=n_adults_range,
#            n_old_adults_range=n_old_adults_range,
#        )
#
#
#class Households(Supergroup):
#    def __init__(self, households: List[Household]):
#        super().__init__()
#        self.members = households
#
#    @classmethod
#    def from_household_compositions(cls, household_compositions: dict):
#        """
#        Initializes households from a dictionary of household compositions.
#
#        Example
#        -------
#        compositions = {
#            "single": {1: {"old": True, "number": 10}, 2: {"old": False, "number": 5}},
#        }
#        households = Households.from_household_compositions(compositions)
#
#        will create 10 households for single old people, and 5 households for non-old
#        single people.
#        """
#        households = []
#        for composition_type, composition_dict in household_compositions.items():
#            composition_type_class_name = (
#                "Household" + composition_type[0].upper() + composition_type[1:]
#            )
#            household_class = _str2class(composition_type_class_name)
#            if "number" in composition_dict:
#                number = composition_dict["number"]
#                composition_parsed = _parse_composition_config(composition_dict)
#                for _ in range(number):
#                    households.append(household_class(**composition_parsed))
#            else:
#                for config_number in composition_dict:
#                    number = composition_dict[config_number]["number"]
#                    composition_parsed = _parse_composition_config(
#                        composition_dict[config_number]
#                    )
#                    for _ in range(number):
#                        households.append(household_class(**composition_parsed))
#        return cls(households)
