import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os


class ValidFreeEnergy:
    """
    Extracts the free energy from an hpbandster run if the run was successful.
    If the free energy of the loss is requested, the negative loss is returned.
    """

    def __init__(self, key):
        """
        :param key: The key of the free energy to extract.
        (one of loss, validation,
        """
        self.key = key

    def __call__(self, run):
        if not self.key == "loss":
            if "info" not in vars(run):
                return None  # broken run
            if run["info"] is None:
                return None  # broken run, but later
            if run["info"][self.key] is None:
                return None  # broken run, but silently
            else:
                return run["info"][self.key]

        else:
            if run[self.key] is None:
                return None
            else:
                return -run[self.key]  # hpbandster minimizes, but we report -loglikelihood


def result_and_runs(path):
    """
    :param path: directory of the results and config json files
    :return: the results and all_runs objects
    """
    result = hpres.logged_results_to_HBS_result(path)
    all_runs = result.get_all_runs()
    return result, all_runs


def sorted_by_value(runs, key="loss"):
    """
    :param runs: outpout of an hpbandster.core.result method
    :param key: the key by which to sort the results
    :return: a sorted list with only valid results
    """

    get_if_valid = ValidFreeEnergy(key)
    return sorted([run for run in runs if get_if_valid(run) is not None], key=get_if_valid)


def print_best(path="", printable="loss", criterion="loss", show_config=True, top_n=10):
    """
    :param path: directory of the results and config json files
    :param printable: value to print
    :param criterion: value to sort by for the top N selection
    :param show_config: Bool prints model config
    :param top_n: number of models to show
    :return:
    """

    result, all_runs = result_and_runs(path)
    id2conf = result.get_id2config_mapping()

    by_criterion = sorted_by_value(all_runs, key=criterion)

    print("Good confs as judged by {}: ".format(criterion))
    for i in range(top_n):

        # get value
        if criterion == "loss":
            value = -by_criterion[::-1][i][printable]  # fix minus from Hpbandster minimization
        else:
            value = by_criterion[::-1][i]["info"][printable]

        # get config id
        id = "".join([(str(id_).rjust(3, " ")) for id_ in by_criterion[::-1][i]["config_id"]])

        # print result
        print(
            "{}. with {}/free energy= {} |id ({})".format(
                str(i + 1).rjust(2), printable, str(round(value, 6)).ljust(12), id
            )
        )

        if show_config:
            config = id2conf[by_criterion[i]["config_id"]]
            print(json.dumps(config, indent=4))


def print_error_configs(path, top_n_broken=10):
    """
    This function picks out and prints the hyperparameters that are most frequent
    in configs that had an interrupted run.
    :param path: directory of the results and config json files
    :param top_n_broken: number of hyperparameters to print
    :return: None
    """
    result, all_runs = result_and_runs(path)
    id2conf = result.get_id2config_mapping()

    all_confs = [id2conf[run["config_id"]] for run in all_runs]
    broken_confs = [id2conf[run["config_id"]] for run in all_runs if run["info"] is None]

    all_hyperparamters_by_usage = Counter(
        [key for conf in all_confs for key in conf["config"].keys()]
    )
    constantly_used = [
        key
        for key in all_hyperparamters_by_usage
        if all_hyperparamters_by_usage[key] == len(all_runs)
    ]
    broken_hyperparams_used = [
        key for conf in broken_confs for key in conf["config"].keys() if key not in constantly_used
    ]

    max_len = max([len(key) for key in broken_hyperparams_used])
    broken_hyperparams_used = [key.ljust(max_len) for key in broken_hyperparams_used]

    print("{} broken runs found".format(len(broken_confs)))
    print("Top {} hyperparams by frequency:".format(top_n_broken))
    temp = np.array(Counter(broken_hyperparams_used).most_common(top_n_broken)).T
    temp = np.array([temp[1], temp[0]]).T
    temp = ["".join(a + ": " + b) for a, b in temp]
    print("\n".join(temp))


def visualize(path):
    """
    This function visualize the behaviour of an hpbandster run
    :param path: directory of the results and config json files
    :return:
    """
    # get results
    result, all_runs = result_and_runs(path)

    # plot:

    # losses by budget
    hpvis.losses_over_time(all_runs)

    # concurent runs over time
    hpvis.concurrent_runs_over_time(all_runs)

    # finished runs over time
    hpvis.finished_runs_over_time(all_runs)

    # spearman rank correlation over budgets
    hpvis.correlation_across_budgets(result)

    # model based configs vs random search
    hpvis.performance_histogram_model_vs_random(all_runs, result.get_id2config_mapping(), show=True)

    plt.show()


if __name__ == "__main__":
    # path = 'results'
    path = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../dynamically_binarized_mnist/results_2")
    )
    print_error_configs(path)
    print("\n")
    print_best(
        path,
        printable="validation accuracy",
        criterion="train accuracy",
        show_config=False,
        top_n=10,
    )
    print("\n")
    print_best(
        path,
        printable="validation accuracy",
        criterion="validation accuracy",
        show_config=True,
        top_n=50,
    )
    print("\n")
    print_best(
        path,
        printable="test accuracy",
        criterion="test accuracy",
        show_config=False,
        top_n=10,
    )
    visualize(path)
