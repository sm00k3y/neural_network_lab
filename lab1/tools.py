import numpy as np
import random


def gen_and_data(DATA_SIZE):
    data = []
    for _ in range(DATA_SIZE):
        data.append(([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)], 1))
        data.append(([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], 0))
        data.append(([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)], 0))
        data.append(([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)], 0))
    return data


def gen_and_data_bi(DATA_SIZE):
    data = []
    for _ in range(DATA_SIZE):
        data.append(([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)], 1))
        data.append(([random.uniform(-0.9, -1.1), random.uniform(-0.9, -1.1)], -1))
        data.append(([random.uniform(-0.9, -1.1), random.uniform(0.9, 1.1)], -1))
        data.append(([random.uniform(0.9, 1.1), random.uniform(-0.9, -1.1)], -1))
    return data
