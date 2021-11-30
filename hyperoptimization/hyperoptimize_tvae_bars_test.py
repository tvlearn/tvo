from argparse import ArgumentParser as Parser

from typing import Tuple

from hyperoptimization.workers import TVAEWorker
from hyperoptimization.utils import parse_hyperopt_args as hyperopt
from hyperoptimization.runs import local_sequential as run


def experiment(parser):

    parser.add_argument("dataset", help="HD5 file as expected in input by tvem.Training")
    parser.add_argument("--Ksize", type=int, default=3, help="size of each K^n set")
    parser.add_argument("--epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument(
        "--net-shape",
        required=True,
        type=parse_net_shape,
        help="column-separated list of layer sizes",
    )
    parser.add_argument("--min_lr", type=float, help="MLP min learning rate", required=True)
    parser.add_argument("--max_lr", type=float, help="MLP max learning rate", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", help="output file for train log", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="seed value for random number generators. default is a random seed",
    )
    return parser


def parse_net_shape(net_shape: str) -> Tuple[int, ...]:
    """
    Parse string with TVAE shape into a tuple.

    :param net_shape: column-separated list of integers, e.g. `"10:10:2"`
    :returns: a tuple with the shape as integers, e.g. `(10,10,2)`
    """
    return tuple(map(int, net_shape.split(":")))


parser = experiment(hyperopt(Parser()))
parsed_args = parser.parse_args()

worker = TVAEWorker
pr = None
# pr = result.logged_results_to_HBS_result("results")
run(worker=worker, parsed_args=parsed_args, previous_run=pr)
