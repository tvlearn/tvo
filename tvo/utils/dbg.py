# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from inspect import currentframe, getframeinfo


def print_location():
    frameinfo = getframeinfo(currentframe().f_back)
    print(frameinfo.filename, frameinfo.function, frameinfo.lineno)
