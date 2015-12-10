#!/usr/bin/env python3

from random import random
from math import log
from abc import ABCMeta, abstractmethod
import numpy as np

class BaseEpidemic:

    __metaclass__ = ABCMeta

    def __init__(self, t0, t_end, N0, s0):
        self._t0     = t0
        self._t_end  = t_end
        self._s0     = s0
        self._N0     = N0
        self._i0     = N0 - s0
        self._tau = []

        self.data = []
        self.rcount = 0
        self.count = 0
        self.bcount = 0

    @abstractmethod
    def get_a(self, s, i):
        pass

    @abstractmethod
    def delta(self, mu, tau, s, i, d):
        pass

    def generate(self):
        t = self._t0
        s = self._s0
        i = self._i0
        d = 0

        self._data = []

        tau = 0

        while i != 0 and t < self._t_end:
            n1 = random()
            n2 = random()

            a = self.get_a(s, i)


            tau = 1 / a[0] * log(1 / n1)

            mu = 0
            sa = 0
            while sa < n2 * a[0]:
                mu += 1
                sa = sa + a[mu]

            t += tau
            s, i, d = self.delta(mu, tau, s, i, d)

            self.data.append([t, s, i, d])


        self.data = np.transpose(np.asarray(self.data))

    def get_data(self):
        return self.data

class EpidemicBasedOnInterval(BaseEpidemic):

    def __init__(self, rates, t0, t_end, N0, s0, death_th):
        super().__init__(t0, t_end, N0, s0)
        self._r_rate = rates[0]
        self._i_rate = rates[1]
        self._death_th = death_th

    def get_a(self, s, i):

        if s == 0:
            return [self._r_rate, 0, self._r_rate]
        else:
            return [self._i_rate + self._r_rate, self._i_rate, self._r_rate]

    def delta(self, mu, tau, s, i, d):
        if mu == 1:
            return s - 1, i + 1, d
        elif mu == 2:
            if tau > self._death_th:
                return s, i - 1, d + 1
            else:
                return s + 1, i - 1, d

class EpidemicBasedOnProbability(BaseEpidemic):

    def __init__(self, rates, t0, t_end, N0, s0, p):
        super().__init__(t0, t_end, N0, s0)
        self._r_rate = rates[0]
        self._i_rate = rates[1]
        self._p = p

    def get_a(self, s, i):

        if s == 0:
            return [self._r_rate, 0, self._r_rate]
        else:
            return [self._i_rate + self._r_rate, self._i_rate, self._r_rate]

    def delta(self, mu, tau, s, i, d):
        if mu == 1:
            return s - 1, i + 1, d
        elif mu == 2:
            n3 = random()
            if n3 > self._p:
                return s, i - 1, d + 1
            else:
                return s + 1, i - 1, d

class SimpleEpidemic(BaseEpidemic):

    def __init__(self, rates, t0, t_end, N0, s0):
        super().__init__(t0, t_end, N0, s0)
        self._r_rate = rates[0]
        self._i_rate = rates[1]
        self._d_rate = rates[2]


    def get_a(self, s, i):
        if s == 0:
            return [self._r_rate + self._d_rate, 0, self._r_rate, self._d_rate]
        else:
            return [self._i_rate + self._r_rate + self._d_rate, self._i_rate, self._r_rate, self._d_rate]

    def delta(self, mu, tau, s, i, d):
        if mu == 1 and s > 0:
            return s - 1, i + 1, d
        elif mu == 2:
            return s + 1, i - 1, d
        elif mu == 3:
            return s, i - 1, d + 1

