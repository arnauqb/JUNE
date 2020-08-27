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
    def test__person_finder(self):
        population = [
            Person.from_attributes(sex="m", age=26, ethnicity="A"),
            Person.from_attributes(sex="f", age=50, ethnicity="A"),
            Person.from_attributes(sex="f", age=50, ethnicity="C"),
        ]
        person_finder = PersonFinder(population)
        p = person_finder(sex="m", age=26, ethnicity="A")
        assert p.sex == "m"
        assert p.age == 26
        assert p.ethnicity == "A"
        p = person_finder(sex="m", age=26, ethnicity="A")
        assert p is None
        p1 = person_finder(sex="f", age=50, ethnicity=None)
        assert p1.ethnicity in ["A", "C"]
        p2 = person_finder(sex="f", age=50, ethnicity=None)
        ll = ["A", "C"]
        ll.remove(p1.ethnicity)
        assert p2.ethnicity in ll
        p3 = person_finder(sex="f", age=50, ethnicity=None)
        assert p3 is None

    def test__find_couple():
        person = Person(sex="m", age=27)






