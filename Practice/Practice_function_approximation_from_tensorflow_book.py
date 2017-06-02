# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:25:18 2017

@author: 14224
"""

import numpy as np
import tensorflow as tf
import math, random
import matplotlib.pyplot as plt

""" contruct data model """
num_points = 1000
np.random.seed(num_points)
function_to_learn = lambda x: np.cos(x) + 0.1*np.random.randn(*x.shape)

""" Network Parameters """
layer_1_neurons = 10

""" parameters """
batch_size = 100
num_epochs = 1500

""" input """
