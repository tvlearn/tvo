import hpbandster.core.result as hpres
from hyperoptimization.runs import from_config as run
from hyperoptimization.workers import TVAEWorker
from hyperoptimization.utils import parse_hyperopt_args as hyperopt
from hyperoptimization.explore import sorted_by_value
from argparse import ArgumentParser as Parser
from typing import Tuple


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

path = ""
result = hpres.logged_results_to_HBS_result(path)
all_runs = result.get_all_runs()
id2conf = result.get_id2config_mapping()

ordered_by_loss = sorted_by_value(all_runs, key="loss")

best_n_configs = 5
if best_n_configs:
    raise Exception("modify to take arbitrary sort key")
for i in range(best_n_configs):
    id = ordered_by_loss[i]["config_id"]
    config = id2conf[id]["config"]
    # config['lr']*=10
    print("Running long experiment with config:")
    print("{:<20} {:<20} ".format("hyperparameter", "value"))
    for key, value in config.items():
        print("{:<20} {:<20} ".format(key, value))

    run(
        config,
        budget=parsed_args.epochs,
        worker=worker,
        parsed_args=parsed_args,
    )
