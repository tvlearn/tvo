# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from coverage import coverage
import sys
assert len(sys.argv) == 2, f'Usage: {sys.argv[0]} <output directory>'

cov = coverage()
cov.load()
cov.html_report(directory=sys.argv[1])
