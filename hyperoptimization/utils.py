def parse_hyperopt_args(parser):
    """
    :param parser: an Argument Parser object from argparse
    :return: non-initialized parser with the necessary hpbandster arguments
    """

    parser.add_argument(
        "--min_budget",
        type=float,
        help="Minimum number of epochs for training.",
        default=1,
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        help="Maximum number of epochs for training.",
        default=5,
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        help="Number of iterations performed by the optimizer",
        default=16,
    )
    parser.add_argument(
        "--worker",
        help="Flag to turn this into a worker process",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="A unique run id for this optimization run. An easy option is "
        "to use the job id of the clusters scheduler.",
        default="derp",
    )
    parser.add_argument(
        "--nic_name",
        type=str,
        help="Which network interface to use for communication.",
        default="lo",
    )
    parser.add_argument(
        "--shared_directory",
        type=str,
        help="A directory that is accessible for all processes, e.g. a NFS share.",
        default=".",
    )

    return parser
