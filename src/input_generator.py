import numpy as np
import signalz


class InputGenerator:

    def __init__(self, start_time, end_time, num_time_steps):
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps 

    def generate_sin(self, amplitude=1.0):
        return np.sin( np.linspace(self.start_time, self.end_time, self.num_time_steps) ) * amplitude

    def generate_mackey_glass(self, a=0.2, b=1, c=0.9, d=17, e=10, initial=0.1):
        return (signalz.mackey_glass(self.num_time_steps+200, a=a, b=b, c=c, d=d, e=e, initial=initial) - 0.8)[200:]