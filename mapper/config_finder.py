# Find the number of GIGA and nano cores compbinations

import numpy as np
import math

import pandas as pd
import itertools
import numpy as np
from itertools import combinations_with_replacement

# Area of each nano and GIGA core (from synthesis)
nano_core_area = 1.2
giga_core_area = 2

max_area = 15

# area = num_nano_cores * nano_core_area + num_giga_cores * giga_core_area

def combinations_permuted(max_area, n_max_cores=10):

    giga = list(range(0, 1, 1))
    nano = list(range(0, 1, 1))
    
    for nano_core in items:
        for giga_core in items:
            area = giga_core * giga_core_area + nano_core * nano_core_area

            if (area < max_area):
                giga_core += 1
                break

        if (area < max_area):
            nano_core += 1

    core_combinations = list(combinations_with_replacement(items, 2))

    return core_combinations

if (__name__ == "__main__"):
    
    core_combination = combinations_permuted(max_area)
    print(core_combination)
    