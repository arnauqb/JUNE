import sys
import infection
import random
import infection as Infection


class Person:
    """
    Primitive version of class person.  This needs to be connected to the full class 
    structure including health and social indices, employment, etc..  The current 
    implementation is only meant to get a simplistic dynamics of social interactions coded.
    
    The logic is the following:
    People can get infected with an Infection, which is characterised by time-dependent
    transmission probabilities and symptom severities (see class descriptions for
    Infection, Transmission, Severity).  The former define the infector part for virus
    transmission, while the latter decide if individuals realise symptoms (we need
    to define a threshold for that).  The symptoms will eventually change the behavior 
    of the person (i.e. intensity and frequency of social contacts), if they need to be 
    treated, hospitalized, plugged into an ICU or even die.  This part of the model is 
    still opaque.   
    
    Since the realization of the infection will be different from person to person, it is
    a characteristic of the person - we will need to allow different parameters describing
    the same functional forms of transmission probability and symptom severity, distributed
    according to a (tunable) parameter distribution.  Currently a non-symmetric Gaussian 
    smearing of 2 sigma around a mean with left-/right-widths is implemented.    
    """
    def __init__(self, person_id, area, age, sex, health_index, econ_index):
        if not self.sane(self, person_id, area, age, sex, health_index, econ_index):
            return
        self.id             = person_id
        self.age            = age
        self.sex            = sex
        self.health_index   = health_index
        self.econ_index     = econ_index
        self.area           = area
        self.household      = None
        self.init_health_information()

    def sane(self, person_id, area, age, sex, health_index, econ_index):
        if (age<0 or age>120 or
            not (sex=="M" or sex=="F") ):
            print ("Error: tried to initialise person with descriptors out of range: ")
            print ("Id = ",person_id," age / sex = ",age,"/",sex)
            print ("economical/health indices: ",econ_index,health_index) 
            sys.exit()
        return True
        
    def name(self):
        return self.person_id

    def age(self):
        return self.age

    def sex(self):
        return self.sex
        
    def health_index(self):
        return self.health_index
    
    def econ_index(self):
        return self.econ_index
    
    def set_household(self,household):
        self.household = household

    def init_health_information():
        self.susceptibility = 1.
        self.healthy        = True
        self.infection      = None
        
    def set_infection(self,infection):
        if (not isinstance(infection, Infection.Infection) and
            not infection==None):
            print ("Error in Infection.Add(",infection,") is not an infection")
            print("--> Exit the code.")
            sys.exit()
        self.infection  = infection
        if not self.infection==None:
            self.healthy = False

    def update_health_status(self,time):
        if (self.infection==None or
            self.infection.still_infected(time) ):
            self.healthy = True
        else:
            self.healthy = False
            
    def is_healthy(self):
        return self.healthy

    def is_infected(self):
        return not self.is_healthy()

    def susceptibility(self):
        return self.susceptibility
    
    def transmission_probability(self,time):
        if self.infection==None:
            return 0.
        return self.infection.transmission_probability(time)

    def symptom_severity(self,time):
        if self.infection==None:
            return 0.
        return self.infection.symptom_severity(time)

    def output(self):
        print ("--------------------------------------------------")
        print ("Person [",self.pname,"]: age = ",self.age," sex = ",self.sex)
        if self.is_healthy():
            print ("-- person is healthy.")
        else:
            print ("-- person is infected.")


    
class Adult(Person):

    def __init__(self, area, age, sex, health_index, econ_index, employed):
        Person.__init__(self, area, age, sex, health_index, econ_index)
        self.employed = employed

class Child(Person):
    def __init__(self, area, age, sex, health_index, econ_index):
        Person.__init__(self, area, age, sex, health_index, econ_index)
