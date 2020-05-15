from june.demography.person import Person
from .abstract import AbstractGroup
from typing import Set, List


class Subgroup(AbstractGroup):
    __slots__ = "_people", "_susceptible", "_infected", "_recovered", "_in_hospital", "_dead"

    def __init__(self):
        """
        A group within a group. For example, children in a household.
        """
        self._people = list()

    def _collate(
            self,
            attribute: str
        ) -> List[Person]:
        collection = list()
        for person in self.people:
            if getattr(person.health_information, attribute):
                collection.append(
                    person
                    )
        return collection

    @property
    def susceptible(self):
        return self._collate('susceptible')

    @property
    def infected(self):
        return self._collate('infected')

    @property
    def recovered(self):
        return self._collate('recovered')

    @property
    def in_hospital(self):
        return self._collate('in_hospital')

    def __contains__(self, item):
        return item in self._people

    def __iter__(self):
        return iter(self._people)

    def clear(self):
        self._people = set()

    @property
    def people(self):
        return self._people

    @property
    def contains_people(self) -> bool:
        """
        Whether or not the group contains people.
        """
        return len(self._people) > 0

    def append(self, person: Person):
        """
        Add a person to this group
        """
        self._people.add(person)

    def remove(self, person: Person):
        """
        Remove a person from this group
        """
        self._people.remove(person)

    def __getitem__(self, item):
        return list(self._people)[item]
