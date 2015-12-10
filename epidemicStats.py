#!/bin/env/python3

from epidemic import EpidemicBasedOnProbability, EpidemicBasedOnInterval, SimpleEpidemic
from statistics import mean
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class EpidemicStats:
    __metaclass__ = ABCMeta

    def __init__(self, rates, t0, t_end, N0, s0):

        self._rates = rates
        self._t0 = t0
        self._t_end = t_end
        self._N0 = N0
        self._s0 = s0

        self._distribution_at_t = {}
        self._data = []

        self._STEP_LIMIT = 100
        self._NUMBER_OF_PATHS = 500
        self._NUMBER_OF_MOMENTS = 200

    @abstractmethod
    def generate(self):
        pass

    def find_quantity_at_t(self, i, q, t):
        pairs = {'s': 1, 'i': 2, 'd': 3}
        if t < self._t0 or t > self._t_end:
            raise ValueError('t is out of range')

        if t < self._data[i][0][0]:
           if q == 's':
               return self._s0
           elif q == 'i':
               return self._N0 - self._s0
           else:
               return 0
        elif t >= self._data[i][0][-1]:
           return self._data[i][pairs[q]][-1]

        num = len(self._data[i][0])

        for j in range(num - 1):
            if t >= self._data[i][0][j] and t < self._data[i][0][j+1]:
                return self._data[i][pairs[q]][j]

    def get_quantity_distribution_at_t(self, q, t):
        if (q, t) in self._distribution_at_t:
            return self._distribution_at_t[(q, t)]
        else:
            x = []
            for i in range(self._NUMBER_OF_PATHS):
                x.append(self.find_quantity_at_t(i, q, t))
            self._distribution_at_t[(q, t)] = x
            return x



    def ensemble_mean(self):

        self.t = np.linspace(self._t0, self._t_end, self._NUMBER_OF_MOMENTS)

        self.s = np.vectorize(lambda v: mean(self.get_quantity_distribution_at_t('s', v)))(self.t)
        self.i = np.vectorize(lambda v: mean(self.get_quantity_distribution_at_t('i', v)))(self.t)
        self.d = np.vectorize(lambda v: mean(self.get_quantity_distribution_at_t('d', v)))(self.t)

        plt.plot(self.t, self.s, color = "blue")
        plt.plot(self.t, self.i, color = "green")
        plt.plot(self.t, self.d, color = "red")
        blue_legend = mpatches.Patch(color = "blue", label = "Susceptible")
        green_legend = mpatches.Patch(color = "green", label = "Infected")
        red_legend = mpatches.Patch(color = "red", label = "Dead")
        plt.legend(handles = [blue_legend, green_legend, red_legend], bbox_to_anchor = (1, 0.9))
        plt.xlabel("t")
        plt.ylabel("number")
        plt.show(block = False)


    def single_path(self):
        data = self._data[0]
        plt.plot(data[0], data[1])
        plt.plot(data[0], data[2])
        plt.plot(data[0], data[3])
        plt.show(block = False)



class EpidemicBasedOnIntervalStats(EpidemicStats):

    def __init__(self, rates, t0, t_end, N0, s0, death_th):
        super().__init__(rates, t0, t_end, N0, s0)
        self._death_th = death_th

    def generate(self):
        self._data = []
        self._distribution_at_t = {}
        for i in range(self._NUMBER_OF_PATHS):
            while True:
                epidemic = EpidemicBasedOnInterval(self._rates, self._t0, self._t_end, self._N0, self._s0, self._death_th)
                epidemic.generate()
                if len(epidemic.get_data()[0]) > self._STEP_LIMIT:
                    break
            data = epidemic.get_data()
            self._data.append(data)


    def extinction_vs_threshold(self):

        for threshold in range(5, 10):
            s = 0
            t_end = -500 + threshold * 1000 # empirical rule, subject to change when death_th is larger
            self._t_end = t_end
            self._death_th = threshold
            self.generate()
            for d in self._data:
                if d[1][-1] == 0 and d[2][-1] == 0:
                    s += 1
            print("Threshold:" + str(threshold) + " Extinction probability is " + str(s / self._NUMBER_OF_PATHS))

    def timescale_vs_threshold(self):

        for threshold in range(3, 10):
            t_end = -500 + threshold * 1000 # empirical rule, subject to change when death_th is larger
            self._death_th = threshold
            self._t_end = t_end
            self.generate()

            t = np.linspace(self._t0, t_end, self._NUMBER_OF_MOMENTS)
            infected = np.vectorize(lambda v: mean(self.get_quantity_distribution_at_t('i', v)))(t)
            for idx, val in enumerate(infected):
                if val < 0.5:
                    print("Threshold: " + str(threshold) + " Timescale is " + str(t[idx]))
                    break


class EpidemicBasedOnProbabilityStats(EpidemicStats):

    def __init__(self, rates, t0, t_end, N0, s0, p):
        super().__init__(rates, t0, t_end, N0, s0)
        self._p = p

    def generate(self):
        self._data = []
        self._distribution_at_t = {}
        for i in range(self._NUMBER_OF_PATHS):
            while True:
                epidemic = EpidemicBasedOnProbability(self._rates, self._t0, self._t_end, self._N0, self._s0, self._p)
                epidemic.generate()
                if len(epidemic.get_data()[0]) > self._STEP_LIMIT:
                    break
            data = epidemic.get_data()
            self._data.append(data)

class SimpleEpidemicStats(EpidemicStats):

    def generate(self):
        self._data = []
        self._distribution_at_t = {}
        for i in range(self._NUMBER_OF_PATHS):
            while True:
                epidemic = SimpleEpidemic(self._rates, self._t0, self._t_end, self._N0, self._s0)
                epidemic.generate()
                if len(epidemic.get_data()[0]) > self._STEP_LIMIT:
                    break
            data = epidemic.get_data()
            self._data.append(data)


if __name__ == '__main__':

    input("Epidemic Based On Interval. Press enter to continue.")
    e1 = EpidemicBasedOnIntervalStats([0.1, 0.15], 0, 7000, 100, 99, 10)
    e1.generate()
    e1.ensemble_mean()
    input("Timescale vs threshold. Press enter to continue.")
    e1.timescale_vs_threshold()
    input("Extinction vs threshold. Press enter to continue.")
    e1.extinction_vs_threshold()

    input("Epidemic Based On Probability. Press enter to continue.")
    e2 = EpidemicBasedOnIntervalStats([0.1, 0.15], 0, 16000, 100, 99, 0.9371)
    e2.generate()
    e2.ensemble_mean()

    input("Simple Epidemic. Press enter to continue.")
    e3 = EpidemicBasedOnIntervalStats([0.1, 0.15 * 0.9371, 0.15 * 0.0629], 0, 16000, 100, 99)
    e3.generate()
    e3.ensemble_mean()


