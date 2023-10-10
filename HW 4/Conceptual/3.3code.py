#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:32:33 2022

@author: ziggy
"""
import math

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def calculate_error(epsilon, N):
    half = (N + 1) // 2
    # for i in range((half,  N + 1):
    sum = 0
    for i in range(half, N + 1):
        curP = nCr(N, i) * math.pow(epsilon, i) * math.pow(1 - epsilon, N - i)
        # print(nCr(N, i) * math.pow(epsilon, i) * math.pow(1 - epsilon, N - i))
        sum += curP
    return sum
        
error = calculate_error(0.4, 5)
print(error) # 0.3174400000000001

error = calculate_error(0.4, 11)
print(error) # 0.24650186752000006

error = calculate_error(0.4, 21)
print(error) # 0.17437786636177277

error = calculate_error(0.6, 5)
print(error) # 0.68256

error = calculate_error(0.6, 11)
print(error) # 0.7534981324799999

error = calculate_error(0.6, 21)
print(error) # 0.8256221336382271
