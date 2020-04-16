import numpy as np
from random import uniform
from scipy import stats
import warnings

"""
This file contains routines to attribute people with different characteristics
according to census data.
"""


class CompanyDistributor:
    """
    Distributes people to different companies
    """

    def __init__(self, companies, msoarea):
        self.msoarea = msoarea
        self.companies_all = companies
        # gather call companies in a given msoarea
        self.companies_msoarea = []
        for company in self.companies_all.members:
            if company.msoa == msoarea:
                self.companies_msoarea.append(company)

    def _randomly_sample_company(self):
        index = np.random.randint(len(self.companies_msoarea))
        return self.companies_msoarea[index]

    def distribute_adults_to_companies(self):
        STUDENT_THRESHOLD = area.world.config["people"]["student_age_group"]
        ADULT_THRESHOLD = area.world.config["people"]["adult_threshold"]
        OLD_THRESHOLD = area.world.config["people"]["old_threshold"]
        
        # this assumes that self.msoarea.people.values() gives the people who WORK in that area
        for person in self.msoarea.work_people:
            if (
                    person.age <= OLD_THRESHOLD # if we already assume the first comment, this seems redundant
                and person.age >= STUDENT_THRESHOLD 
            ):  # person age from 20 up to 74 yo
                person_industry = person.industry
                assigned = False
                while assigned == False:
                    # randomly sample from companies in msoarea rather than filling from the start
                    # as not all companies will be filled and there may be a mismatch in the number of people
                    # and the number of companies
                    company = self._randomly_sample_company()
                    if company.n_employees == company.n_employees_max:
                            pass
                    else:
                        company.n_employees +=1
                        assigned = True
