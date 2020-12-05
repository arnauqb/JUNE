import yaml
from random import shuffle, randint

from june.groups.leisure import SocialVenueDistributor
from june.paths import configs_path

default_config_filename = configs_path / "defaults/groups/leisure/visits.yaml"


class ResidenceVisitsDistributor(SocialVenueDistributor):
    """
    This is a social distributor specific to model visits between residences,
    ie, visits between households or to care homes. The meaning of the parameters
    is the same as for the SVD. Residence visits are not decied on neighbours or distances
    so we ignore some parameters.
    """

    def __init__(self, times_per_week, hours_per_day, drags_household_probability=0):
        super().__init__(
            social_venues=None,
            times_per_week=times_per_week,
            hours_per_day=hours_per_day,
            drags_household_probability=drags_household_probability,
            neighbours_to_consider=None,
            maximum_distance=None,
            leisure_subgroup_type=None,
        )

    @classmethod
    def from_config(cls, config_filename: str = default_config_filename):
        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)

    def link_households_to_households(self, super_areas):
        """
        Links people between households. Strategy: We pair each household with 0, 1,
        or 2 other households (with equal prob.). The household of the former then
        has a probability of visiting the household of the later
        at every time step.

        Parameters
        ----------
        super_areas
            list of super areas
        """
        for super_area in super_areas:
            households_super_area = []
            for area in super_area.areas:
                households_super_area += [
                    household
                    for household in area.households
                    if household.type != "communal"
                ]
                shuffle(households_super_area)
            for household in households_super_area:
                if household.size == 0:
                    continue
                households_to_link_n = randint(1, 3)
                households_to_visit = []
                n_linked = 0
                while n_linked < households_to_link_n:
                    house_idx = randint(0, len(households_super_area) - 1)
                    house = households_super_area[house_idx]
                    if house.id == household.id:
                        continue
                    if not house.people:
                        continue
                    households_to_visit.append(house)
                    n_linked += 1
                if households_to_visit:
                    household.residences_to_visit = tuple(
                        *household.residences_to_visit, households_to_visit
                    )

    def link_households_to_care_homes(self, super_areas):
        """
        Links households and care homes in the giving super areas. For each care home,
        we find a random house in the super area and link it to it.
        The house needs to be occupied by a family, or a couple.

        Parameters
        ----------
        super_areas
            list of super areas
        """
        for super_area in super_areas:
            households_super_area = []
            for area in super_area.areas:
                households_super_area += [
                    household
                    for household in area.households
                    if household.type in ["families", "ya_parents", "nokids"]
                ]
            shuffle(households_super_area)
            for area in super_area.areas:
                if area.care_home is not None:
                    people_in_care_home = [
                        person for person in area.care_home.residents
                    ]
                    for i, person in enumerate(people_in_care_home):
                        household = households_super_area[i]
                        household.residences_to_visit = (
                            *household.residences_to_visit,
                            area.care_home,
                        )

    def get_leisure_group(self, person):
        candidates = person.residence.group.residences_to_visit
        n_candidates = len(candidates)
        if n_candidates == 0:
            return
        elif n_candidates == 1:
            group = candidates[0]
        else:
            group = candidates[randint(0, n_candidates - 1)]
        return group

    def get_poisson_parameter(self, sex, age, day_type, working_hours):
        """
        This differs from the super() implementation in that we do not allow
        visits during working hours as most people are away.
        """
        if working_hours:
            return 0
        poisson_parameter = self.poisson_parameters[day_type][sex][age]
        return poisson_parameter
