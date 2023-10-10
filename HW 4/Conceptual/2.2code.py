#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:43:52 2022

@author: ziggy
"""

# from mpl_toolkits import mplot3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
x = np.array([2,2,3,1,1,1])
y = np.array([3,4,1,1,2,1])
z = np.array([1,0,1,0,1,1])

ax.scatter(2, 3, 1, color = "red")
ax.scatter(2, 4, 0, color = "red")
ax.scatter(1, 1, 1, color = "green")
ax.scatter(3, 1, 1, color = "blue")
ax.scatter(1, 1, 0, color = "blue")
ax.scatter(1, 2, 1, color = "blue")

