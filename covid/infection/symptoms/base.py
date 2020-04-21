from covid.parameters import ParameterInitializer
import numpy as np
import random

ALLOWED_SYMPTOM_TAGS = [
    "asymptomatic",
    "influenza-like illness",
    "pneumonia",
    "hospitalised",
    "intensive care",
    "dead",
]


class Symptoms(ParameterInitializer):
    def __init__(self, timer, health_index, user_parameters, required_parameters):
        super().__init__("symptoms", required_parameters)
        self.initialize_parameters(user_parameters)
        self.timer = timer
        self.infection_start_time = self.timer.now
        self.last_time_updated = self.timer.now  # for testing
        self.health_index = health_index
        self.maxseverity = random.random()
        self.tags = ALLOWED_SYMPTOM_TAGS
        self.severity = 0.0

    def update_severity(self):
        self.last_time_updated = self.timer.now
        pass

    @property
    def n_tags(self):
        return len(self.tags)

    @property
    def tag(self):
        return self.fix_tag()

    def fix_tag(self):
        if self.severity <= 0.0:
            return "healthy"
        index = np.searchsorted(self.health_index, self.severity)
        return self.tags[index + 1]
