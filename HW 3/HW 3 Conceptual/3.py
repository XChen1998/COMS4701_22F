#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:07:05 2022

@author: ziggy
"""
import numpy as np
def entropy(x):
    return -x * np.log2(x)


print(entropy(10.0/16.0))
print(entropy(6/16.0))

print(entropy(10.0/16.0) + entropy(6/16.0))

f1_1 = (entropy(0.1) + entropy(0.9)) * 10.0/16.0
f1_2 = (entropy(1.0 / 6.0) + entropy(5.0 / 6.0)) * 6.0/16.0
print(f1_1 + f1_2)

f2_1 = (entropy(0.000000000000000000001) + entropy(1)) * 4.0 / 16.0
# print(f2_1)
f2_2 = (entropy(6.0 / 12.0) + entropy(6.0 / 12.0)) * 12.0 / 16.0
print(f2_1 + f2_2)