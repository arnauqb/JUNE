class Transmission:

    def __init__(self):

        self.probability = 0.0

    def update_probability_from_delta_time(self, time):
        raise NotImplementedError()

class TransmissionConstant(Transmission):

    def __init__(self, probability=0.3):

        super().__init__()

        self.probability = probability

    def update_probability_from_delta_time(self, delta_time):

        pass