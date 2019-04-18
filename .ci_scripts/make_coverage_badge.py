# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from coverage import coverage
import sys
from urllib.request import Request, urlopen
assert len(sys.argv) == 2, f'Usage: {sys.argv[0]} <output filename>'


def pick_color(coverage):
    colors = ((95, 'brightgreen'), (90, 'green'), (70, 'yellow'), (40, 'orange'), (0, 'red'))
    for perc, color in colors:
        if coverage >= perc:
            return color


cov = coverage()
cov.load()
total = int(cov.report())
color = pick_color(total)

badge_url = f'https://img.shields.io/badge/coverage-{total}%25-{color}.svg?style=flat-square'
# shields.io blocks the urllib user agent
req = Request(badge_url, headers={'User-Agent': 'Mozilla/5.0'})
badge_img = urlopen(req).read()
with open(sys.argv[1], 'wb') as out_file:
    out_file.write(badge_img)
