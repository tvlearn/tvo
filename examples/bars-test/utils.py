# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import sys
import torch.distributed as dist
from typing import Tuple

from tvo import get_run_policy
from tvo.utils.parallel import init_processes as _init_processes


def init_processes() -> Tuple[int, int]:
    if get_run_policy() == "seq":
        return 0, 1
    else:
        assert get_run_policy() == "mpi"
        _init_processes()
        return dist.get_rank(), dist.get_world_size()


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
