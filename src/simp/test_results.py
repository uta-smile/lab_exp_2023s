#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Mar 19, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : test_results.py
project  : lab_exp_2023s
license  : GPL-3.0+

Test results pred.txt
"""
# Utils
from rich import print

# Math
import numpy as np

pred = np.loadtxt("labels.csv", delimiter=",", skiprows=1)
label = np.load("test.y.npz")["y"]

print("Accuracy: ", np.mean(pred[:, 1] == label))
