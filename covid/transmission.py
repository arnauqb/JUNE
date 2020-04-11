import numpy as np
import random


class Transmission:
    """
    The probability for an individual to transmit the infection.
    This is time-dependent, and the actual value is calculated in the method
    Probability.  We allow to vary parameters around their mean value with
    a left- and right-sided Gaussian described by sigma and the result
    limited by 2 sigma in either direction or physical limits.  

    Currently two forms are implemented:
    - TransmissionSI
    a constant transmission probability, given by the value, with infinite length
    The only parameter is probability
    - TransmissionConstantInterval
    a constant transmission probability, given by the value and the length
    of the transmission period.
    Parameters are probability and end_time
    - TransmissionXNExp
    a probablity of the form $P(t) = P_0 x^n exp(-x/a)$ with parameters given
    by P_0 = probability, n = exponent, and a = norm 

    TODO: we should try and map this onto the Flute/Imperial models, as far
    as possible, to have a baseline and to facilitate validation.
    """

    def __init__(self, person, params={}, time=-1.0):
        self.person = person
        self.starttime = time
        self.value = 0.0
        self.init(params)

    def init(self, params):
        pass

    def probability(self, time):
        if time >= self.starttime:
            self.calculate(time)
        else:
            self.value = 0.0
        return max(0.0, self.value)

    def calculate(self, time):
        pass


#################################################################################
#################################################################################
#################################################################################


class TransmissionSI(Transmission):
    def init(self, params):
        self.prob = min(1.0, max(0.0, params["probability"]["value"]))

    def calculate(self, time):
        self.value = self.prob


#################################################################################
#################################################################################
#################################################################################


class TransmissionSIR(Transmission):
    def init(self, params):
        self.probT = min(1.0, max(0.0, params["probability"]["value"]))
        self.probR = min(1.0, max(0.0, params["recovery"]["value"]))
        self.RecoverCutoff = params["recovery_cutoff"]["value"]
        self.lasttime = self.starttime  # last time they had a chance to recover

    def calculate(self, time):
        if self.probR != 0:
            if self.probT > 0 and time > self.lasttime:
                for t in range(self.lasttime, time, 1):
                    if random.random() < self.probR:
                        self.person.set_susceptibility(0)  # immune
                        self.person.set_recovered(True)
                        break  # can only recover once
        else:
            if self.probT > 0 and time > self.lasttime:
                for t in range(self.lasttime, time, 1):
                    if time > self.starttime + self.RecoverCutoff:
                        self.person.set_susceptibility(0)  # immune
                        self.person.set_recovered(True)
                        break  # can only recover once
        self.lasttime = time  # update last time
        self.value = self.probT


#################################################################################
#################################################################################
#################################################################################


class TransmissionConstantInterval(Transmission):
    def init(self, params):
        self.prob = min(1.0, max(0.0, params["probability"]["value"]))
        self.endtime = params["end_time"]["value"]
        self.value = 0

    def calculate(self, time):
        if time <= self.starttime + self.endtime:
            self.value = self.prob
        else:
            self.value = 0.0


#################################################################################
#################################################################################
#################################################################################


class TransmissionXNExp(Transmission):
    def init(self, params):
        self.prob = min(1.0, max(0.0, params["probability"]["value"]))
        self.exponent = max(0.0, params["exponent"]["value"])
        self.tailfactor = params["norm"]["value"]
        if self.tailfactor < 0.001:
            self.tailfactor = 0.001
        self.init_norm

    def init_norm(self):
        x0 = self.exponent * self.tailfactor
        self.norm = x0 ** self.exponent * np.exp(-x0 / self.tailfactor)
        self.norm = 1.0 / self.norm

    def calculate(self, time):
        dt = time - self.starttime
        self.value = (
            self.norm * self.prob * dt ** self.exponent * np.exp(-dt / self.tailfactor)
        )
