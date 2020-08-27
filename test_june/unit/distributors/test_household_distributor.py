import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pytest

from june.demography.person import Person
from june.demography import Demography
from june.demography.geography import Geography
from june.groups import Household, Households
from june.distributors import HouseholdDistributor, PersonFinder


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
