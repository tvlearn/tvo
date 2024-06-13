import os
import pickle

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB


def from_config(config, budget, worker, parsed_args, *args, **kwargs):

    host = hpns.nic_name_to_host(parsed_args.nic_name)

    # Start a nameserver:
    NS = hpns.NameServer(
        run_id=parsed_args.run_id,
        host=host,
        port=None,
        working_directory=parsed_args.shared_directory,
    )
    ns_host, ns_port = NS.start()

    # Start local worker
    w = worker(
        run_id=parsed_args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        timeout=120,
        parsed_args=parsed_args,
        *args,
        **kwargs
    )
    w.run(background=True)
    res = w.compute(config=config, budget=budget, working_directory=os.getcwd(), *args, **kwargs)
    with open(os.path.join(parsed_args.shared_directory, "results.pkl"), "wb") as fh:
        pickle.dump(res, fh)

    NS.shutdown()


def local_sequential(worker, parsed_args, previous_run=None, *args, **kwargs):

    # get hostname
    host = hpns.nic_name_to_host(parsed_args.nic_name)

    # log results
    result_logger = hpres.json_result_logger(directory=parsed_args.shared_directory, overwrite=True)

    # Start a nameserver:
    NS = hpns.NameServer(
        run_id=parsed_args.run_id,
        host=host,
        port=None,
        working_directory=parsed_args.shared_directory,
    )
    ns_host, ns_port = NS.start()

    # Start local worker
    w = worker(
        run_id=parsed_args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        timeout=120,
        parsed_args=parsed_args,
        *args,
        **kwargs
    )
    w.run(background=True)

    # Run an optimizer
    # previous_run = hpres.logged_results_to_HBS_result('')

    bohb = BOHB(
        configspace=w.get_configspace(),
        run_id=parsed_args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        min_budget=parsed_args.min_budget,
        max_budget=parsed_args.max_budget,
        previous_result=previous_run,
    )
    res = bohb.run(n_iterations=parsed_args.n_iterations)

    # store results
    with open(os.path.join(parsed_args.shared_directory, "results.pkl"), "wb") as fh:
        pickle.dump(res, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


def on_the_cluster(worker, parsed_args, previous_run=None, *args, **kwargs):

    host = hpns.nic_name_to_host(parsed_args.nic_name)

    NS = hpns.NameServer(
        run_id=parsed_args.run_id, host=host, port=0, working_directory=parsed_args.shared_directory
    )
    ns_host, ns_port = NS.start()

    w = worker(
        sleep_interval=0.5,
        run_id=parsed_args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        parsed_args=parsed_args,
    )
    w.run(background=True)

    bohb = BOHB(
        configspace=w.get_configspace(),
        run_id=parsed_args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=parsed_args.min_budget,
        max_budget=parsed_args.max_budget,
    )
    res = bohb.run(n_iterations=parsed_args.n_iterations, min_n_workers=parsed_args.n_workers)

    with open(os.path.join(parsed_args.shared_directory, "results.pkl"), "wb") as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
