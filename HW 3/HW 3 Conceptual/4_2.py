#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:40:00 2022

@author: ziggy
"""
import numpy as np
def fun(x):
    return -8.0 + 1.2*x

x_set = [55, 60, 66, 72, 85, 90]
y_set = [67, 63, 72, 90, 93, 92]

sum = 0

for i, item in enumerate(x_set):
    print(fun(item))
    sum += np.power((fun(item) - y_set[i]), 2)

print(sum)