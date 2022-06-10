import h5py
import numpy as np
import torch as to
from tvo.utils import get
from tvo.models import BernoulliTVAE as TVAE
from tvo.exp import EEMConfig, ExpConfig, Training, Testing

from hyperoptimization.models import FCDeConvNetSigOut as FCDeConvNet

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.INFO)


class BaseWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**self.extract_worker_args(**kwargs))

    def extract_worker_args(self, **kwargs):
        """
        This function enables the keyword arguments dictionary that is passed to the Worker class
        to contain keywords that are not named explicitly by the base Worker class, but are
        otherwise useful to a downstream class, e.g. to the TVAE.
        :param kwargs: any **kwargs
        :return: inputs accepted by Worker class
        """
        assert "run_id" in kwargs.keys(), "run_id is necessary"
        kw = {
            "run_id": None,
            "nameserver": None,
            "nameserver_port": None,
            "logger": None,
            "host": None,
            "id": None,
            "timeout": None,
        }
        for key in kw:
            if key in kwargs:
                kw[key] = kwargs[key]
        return kw


# TODO: See if it is useful to make a cleaner separation between model and worker
class TVAEWorker(BaseWorker):
    def __init__(self, parsed_args, **kwargs):
        """
        :param parsed_args: list of arguments passed to the script. It is expected to
         contain the following:
        - Ksize: number of states to be kept for truncated inference
        - dataset: name of the dataset to be used
        - epochs per half cycle: number of epochs until a half cycle of cyclic learning
          rate is completed
        - batch size: number of samples per batch
        - output: name of the output file
        - min_lr: minimum learning rate for the cyclic learning rate scheduler
        - max_lr: maximum learning rate for the cyclic learning rate scheduler
        - net_shape: shape of the network. If no H argument is passed, the final layer
          is used to infer the H size.
        - H: size of the first generative layer. This option should be used, as net_shape
          will be phased out.
        - cyclic_lr: whether to use cyclic learning rate or not. If False, the learning
          rate will be constant.
        :param kwargs: Additional arguments to be passed to the underlying Worker class.
        """

        # call base class constructor
        super().__init__(**kwargs)

        # extract args
        self.S = S = parsed_args.Ksize
        self.data_fname = parsed_args.dataset
        self.epochs_per_half_cycle = 1
        self.batch_size = parsed_args.batch_size
        self.output = parsed_args.output
        self.min_lr = parsed_args.min_lr
        self.max_lr = parsed_args.max_lr

        # infer size of H from net_shape. TODO: phase net_shape out
        try:
            net_shape = parsed_args.net_shape
            self.H = H = net_shape[-1]
        except AttributeError:
            self.H = H = parsed_args.H

        try:
            self.cyclic_lr = parsed_args.cyclic_lr
        except AttributeError:
            self.cyclic_lr = False

        # infer hyperparameter status
        self.is_hyperparameter_S = not (S)
        self.is_hyperparameter_H = not (H)
        self.is_hyperparameter_EEM = False

        # loads data, sets N and D
        self.handle_data(kwargs)

        # set the config space
        self.set_configspace()

        # print out dataset information
        print(f"\ninput file: {parsed_args.dataset}")
        try:
            print(f"true logL: {self.data_file['ground_truth']['logL'][...]}")
        except KeyError:
            pass

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        :param config: a config in hpbandster style that contains the model hyperparameters.
        :param budget: amount of epochs to run the model
        :param working_directory: arg used by hpbandster
        :param args: other args
        :param kwargs: other keyworded args
        :return: loss and additional run information
        """

        # extract S and H if they are hyperparameters
        self.extract_hypers_from_config(config)

        # define the model
        model = self.get_external_model(config)

        # setup optimizer
        if config["optimizer"] == "SGD":
            optimizer = to.optim.SGD(
                model.parameters(),
                lr=config["lr"],
                momentum=config["sgd_momentum"],
            )
        elif config["optimizer"] == "Adam":
            optimizer = to.optim.Adam(model.parameters(), lr=config["lr"])
        else:
            raise NotImplementedError("Currently we support only SGD with momentum and Adam")

        # todo: fix in TVEM deep tvae branch
        gpu = to.device("cuda:0")
        model.to(gpu)
        model.device = gpu

        # setup TVAE
        cycliclr_half_step_size = np.ceil(self.N / self.batch_size) * self.epochs_per_half_cycle

        if not self.cyclic_lr:
            self.min_lr = self.max_lr = config["lr"]

        model = TVAE(
            external_model=model,
            shape=None,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            cycliclr_step_size_up=cycliclr_half_step_size,
            optimizer=optimizer,
            precision=to.double,
        )

        exp_conf = ExpConfig(
            batch_size=self.batch_size,
            output=self.output,
            data_transform=self.data_transform,
        )
        estep_conf = self.get_EEM_conf(config)

        # setup training
        data_fname = self.data_fname

        training = Training(exp_conf, estep_conf, model, data_fname, self.valid_fname)
        testing = Testing(exp_conf, estep_conf, model, data_fname)
        print("\nlearning...")
        training_results = []
        for train_log in training.run(int(budget)):
            train_log.print()
            training_results.append(train_log._results)

        testing_results = []
        for test_log in testing.run(1):
            # test_log.print()
            testing_results.append(test_log._results)

        train_F, subs = get(training_results[-1], "train_F", "train_subs")
        valid_F, subs = get(training_results[-1], "test_F", "test_subs")

        test_F, subs = get(testing_results[-1], "test_F", "test_subs")

        # optimizable = -train_F  # HpBandSter always minimizes
        #
        # if self.valid_fname:
        #     optimizable = -valid_F  # HpBandSter always minimizes

        return {
            "loss": -train_F if -train_F else np.nan,
            "info": {
                "test accuracy": test_F,
                "train accuracy": train_F,
                "validation accuracy": valid_F,
                "number of parameters": model._external_model.number_of_parameters(),
            },
        }

    def get_external_model(self, config):
        # unpack external model args from config
        (
            n_deconv_layers,
            n_fc_layers,
            W_shapes,
            fc_activations,
            dropouts,
            dc_activations,
            n_filters,
            batch_norms,
            dropout_rate,
            kernels,
        ) = self.model_args_from_(config, sanity_checks=False)

        # setup external model
        model = FCDeConvNet(
            n_deconv_layers=n_deconv_layers,
            n_fc_layers=n_fc_layers,
            W_shapes=W_shapes,
            fc_activations=fc_activations,
            dc_activations=dc_activations,
            n_filters=n_filters,
            batch_norms=batch_norms,
            dropouts=dropouts,
            dropout_rate=dropout_rate,
            input_size=self.H,
            output_shape=self.D,
            filters_from_fc=1,
            kernels=kernels,
        )
        model.H0 = model.shape[0]
        model.D = self.D
        model.double()
        return model

    def get_EEM_conf(self, config):
        if self.is_hyperparameter_EEM:
            estep_conf = EEMConfig(
                n_states=config["S"],
                n_parents=config["n_parents"],
                n_children=config["n_children"],
                n_generations=1,
                crossover=False,
            )
        else:
            estep_conf = EEMConfig(
                n_states=self.S,
                n_parents=min(3, self.S),
                n_children=min(2, self.S),
                n_generations=1,
                crossover=False,
            )
        return estep_conf

    def set_configspace(self):
        add_EEM = self.add_EEM
        add_fc_deconv = self.add_FCDeconv

        if self.is_hyperparameter_H and self.is_hyperparameter_EEM:

            def custom_configspace():
                cs = CS.ConfigurationSpace()
                cs = add_EEM(add_fc_deconv(cs))
                H = CSH.UniformIntegerHyperparameter(name="H", lower=1, upper=10)
                cs.add_hyperparameters([H])
                return cs

        elif self.is_hyperparameter_H:

            def custom_configspace():
                cs = CS.ConfigurationSpace()
                cs = add_fc_deconv(cs)
                H = CSH.UniformIntegerHyperparameter(name="H", lower=1, upper=10)
                cs.add_hyperparameters([H])
                return cs

        elif self.is_hyperparameter_EEM:

            def custom_configspace():
                cs = CS.ConfigurationSpace()
                cs = add_EEM(add_fc_deconv(cs))
                return cs

        else:

            add_fc_deconv = self.add_FCDeconv

            def custom_configspace():
                cs = CS.ConfigurationSpace()
                return add_fc_deconv(cs)

        self.get_configspace = custom_configspace  # .__get__(custom_configspace)

    @staticmethod
    def add_EEM(cs):
        n_states = CSH.UniformIntegerHyperparameter(name="S", lower=1, upper=6)
        n_parents = CSH.UniformIntegerHyperparameter(name="n_parents", lower=1, upper=n_states)
        n_children = CSH.UniformIntegerHyperparameter(
            name="n_children", lower=1, upper=max(n_parents, 2)
        )
        cs.add_hyperparameters([n_states, n_parents, n_children])
        return cs

    @staticmethod
    def add_FCDeconv(cs):
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle
        categorical input parameter.
        :return: ConfigurationsSpace-Object
        """

        lr = CSH.UniformFloatHyperparameter(
            "lr", lower=1e-6, upper=1e-1, default_value="1e-2", log=True
        )

        # setup optimizers
        optimizer = CSH.CategoricalHyperparameter("optimizer", ["Adam", "SGD"])
        sgd_momentum = CSH.UniformFloatHyperparameter(
            "sgd_momentum", lower=0.0, upper=0.99, default_value=0.9, log=False
        )
        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")

        cs.add_condition(cond)

        # set general block length
        max_block_length = 5  # maximum block size for deconv or linear stack

        # define the  linear blocks
        num_linear_layers = CSH.UniformIntegerHyperparameter(
            "num_linear_layers",
            lower=1,
            upper=max_block_length,
            default_value=2,
            log=False,
        )
        root_W_shapes = CSH.UniformIntegerHyperparameter(
            "root_W_shapes", lower=7, upper=14, default_value=8, log=False
        )  # squared in the compute function as of August 2021

        # define positions of dropout layers
        has_dropout_1 = CSH.UniformIntegerHyperparameter(
            "dropout_1", lower=0, upper=1, default_value=0, log=False
        )
        has_dropout_2 = CSH.UniformIntegerHyperparameter(
            "dropout_2", lower=0, upper=1, default_value=0, log=False
        )
        has_dropout_3 = CSH.UniformIntegerHyperparameter(
            "dropout_3", lower=0, upper=1, default_value=0, log=False
        )
        has_dropout_4 = CSH.UniformIntegerHyperparameter(
            "dropout_4", lower=0, upper=1, default_value=0, log=False
        )
        has_dropout_5 = CSH.UniformIntegerHyperparameter(
            "dropout_5", lower=0, upper=1, default_value=0, log=False
        )

        # define activations for the fully connected stack
        activation_list = ["nn.Tanh", "nn.Sigmoid", "nn.LeakyReLU"]
        activation_list = [activation_list[-1]]
        fc_activation_1 = CSH.CategoricalHyperparameter("fc_activation_1", ["nn.Tanh"])
        fc_activation_2 = CSH.CategoricalHyperparameter("fc_activation_2", activation_list)
        fc_activation_3 = CSH.CategoricalHyperparameter("fc_activation_3", activation_list)
        fc_activation_4 = CSH.CategoricalHyperparameter("fc_activation_4", activation_list)
        fc_activation_5 = CSH.CategoricalHyperparameter("fc_activation_5", activation_list)

        # define the deconv blocks
        num_deconv_layers = CSH.UniformIntegerHyperparameter(
            "num_deconv_layers",
            lower=0,
            upper=max_block_length,
            default_value=2,
        )

        # define filter ranges
        # Todo: take filter dimensionality from x for the final filter
        # TODO: remove last filter from hyperparameters

        num_filters_1 = CSH.CategoricalHyperparameter("num_filters_1", [1, "1"])
        num_filters_2 = CSH.UniformIntegerHyperparameter(
            "num_filters_2", lower=1, upper=12, default_value=4, log=True
        )
        num_filters_3 = CSH.UniformIntegerHyperparameter(
            "num_filters_3", lower=1, upper=12, default_value=4, log=True
        )
        num_filters_4 = CSH.UniformIntegerHyperparameter(
            "num_filters_4", lower=1, upper=12, default_value=4, log=True
        )
        num_filters_5 = CSH.UniformIntegerHyperparameter(
            "num_filters_5", lower=1, upper=12, default_value=4, log=True
        )

        # define existence of per-layer batch normalization
        has_batch_norm_1 = CSH.UniformIntegerHyperparameter(
            "batch_norm_1", lower=0, upper=1, default_value=0, log=False
        )
        has_batch_norm_2 = CSH.UniformIntegerHyperparameter(
            "batch_norm_2", lower=0, upper=1, default_value=0, log=False
        )
        has_batch_norm_3 = CSH.UniformIntegerHyperparameter(
            "batch_norm_3", lower=0, upper=1, default_value=0, log=False
        )
        has_batch_norm_4 = CSH.UniformIntegerHyperparameter(
            "batch_norm_4", lower=0, upper=1, default_value=0, log=False
        )
        has_batch_norm_5 = CSH.UniformIntegerHyperparameter(
            "batch_norm_5", lower=0, upper=1, default_value=0, log=False
        )

        # define activations for the deconv stack
        activation_list = ["nn.Tanh", "nn.Sigmoid", "nn.LeakyReLU"]
        activation_list = [activation_list[-1]]
        # raise ArithmeticError
        dc_activation_1 = CSH.CategoricalHyperparameter("dc_activation_1", ["nn.Tanh"])
        dc_activation_2 = CSH.CategoricalHyperparameter("dc_activation_2", activation_list)
        dc_activation_3 = CSH.CategoricalHyperparameter("dc_activation_3", activation_list)
        dc_activation_4 = CSH.CategoricalHyperparameter("dc_activation_4", activation_list)
        dc_activation_5 = CSH.CategoricalHyperparameter("dc_activation_5", activation_list)

        # add fc hyperparams
        cs.add_hyperparameters(
            [
                num_linear_layers,
                root_W_shapes,
                has_dropout_1,
                has_dropout_2,
                has_dropout_3,
                has_dropout_4,
                has_dropout_5,
                fc_activation_1,
                fc_activation_2,
                fc_activation_3,
                fc_activation_4,
                fc_activation_5,
            ]
        )
        # add dc hyperparameters
        cs.add_hyperparameters(
            [
                num_deconv_layers,
                num_filters_1,
                num_filters_2,
                num_filters_3,
                num_filters_4,
                num_filters_5,
                has_batch_norm_1,
                has_batch_norm_2,
                has_batch_norm_3,
                has_batch_norm_4,
                has_batch_norm_5,
                dc_activation_1,
                dc_activation_2,
                dc_activation_3,
                dc_activation_4,
                dc_activation_5,
            ]
        )

        # Add conditions to hyperparameters.
        # Activate deeper hyperparameters only if their corresponding layer is present

        # fully connected stack
        for i in range(2, max_block_length + 1):
            dropout_cond = CS.GreaterThanCondition(
                eval("has_dropout_{}".format(i)), num_linear_layers, i - 1
            )
            cs.add_condition(dropout_cond)
            activation_cond = CS.GreaterThanCondition(
                eval("fc_activation_{}".format(i)), num_linear_layers, i - 1
            )
            cs.add_condition(activation_cond)

        # deconv stack
        for i in range(1, max_block_length + 1):
            batch_norm_cond = CS.GreaterThanCondition(
                eval("has_batch_norm_{}".format(i)), num_deconv_layers, i - 1
            )
            cs.add_condition(batch_norm_cond)
            activation_cond = CS.GreaterThanCondition(
                eval("dc_activation_{}".format(i)), num_deconv_layers, i - 1
            )
            cs.add_condition(activation_cond)
            filter_cond = CS.GreaterThanCondition(
                eval("num_filters_{}".format(i)), num_deconv_layers, i - 1
            )
            cs.add_condition(filter_cond)

        # set global dropout rate
        dropout_rate = CSH.UniformFloatHyperparameter(
            "dropout_rate", lower=0.0, upper=0.9, default_value=0.5, log=False
        )
        cs.add_hyperparameters([dropout_rate])

        return cs

    def model_args_from_(self, config, sanity_checks=False):

        # unpack values from hpbandster config
        if self.is_hyperparameter_S:
            self.S = config["S"]
        if self.is_hyperparameter_H:
            self.H = config["H"]

        n_deconv_layers = config["num_deconv_layers"]
        n_fc_layers = config["num_linear_layers"]

        W_shapes = config["root_W_shapes"] ** 2

        if type(W_shapes) is not list:
            assert type(W_shapes) is int
            W_shapes = [W_shapes for _ in range(n_fc_layers)]

        fc_activations = [config["fc_activation_{}".format(i + 1)] for i in range(n_fc_layers)]
        dropouts = [config["dropout_{}".format(i + 1)] for i in range(n_fc_layers)]
        dc_activations = [config["dc_activation_{}".format(i + 1)] for i in range(n_deconv_layers)]
        n_filters = [config["num_filters_{}".format(i + 1)] for i in range(n_deconv_layers)]
        batch_norms = [
            config["batch_norm_{}".format(i + 1)] if "batch_norm_1" in config.keys() else 0
            for i in range(n_deconv_layers)
        ]
        dropout_rate = config["dropout_rate"]
        kernels = [
            config["num_kernels_{}".format(i + 1)]
            for i in range(n_deconv_layers)
            if "num_kernels_1" in config.keys()
        ]
        if sanity_checks:
            # todo: use in testing

            # check expected argument length
            assert len(fc_activations) == len(
                [config[key] for key in config if "fc_activation_" in key]
            )
            assert len(dropouts) == len(
                [config[key] for key in config if "dropout_" in key and "rate" not in key]
            )
            assert len(dc_activations) == len(
                [config[key] for key in config if "dc_activation_" in key]
            )
            assert len(n_filters) == len([config[key] for key in config if "num_filters_" in key])
            assert len(batch_norms) == len([config[key] for key in config if "batch_norm_" in key])

            # if len(n_filters):
            #     assert n_filters[0] == 1

            # check expected argument type
            assert type(n_fc_layers) is int
            assert type(n_deconv_layers) is int

            for a in fc_activations:
                assert type(a) is str
            for dr in dropouts:
                assert dr in [0, 1]
            for a in dc_activations:
                assert type(a) is str

            for f in n_filters:
                assert type(f) is int

            for bn in batch_norms:
                assert bn in [0, 1]

            assert 1 >= dropout_rate >= 0

        return (
            n_deconv_layers,
            n_fc_layers,
            W_shapes,
            fc_activations,
            dropouts,
            dc_activations,
            n_filters,
            batch_norms,
            dropout_rate,
            kernels,
        )

    def handle_data(self, **kwargs):
        # extract data from file
        self.data_file = h5py.File(self.data_fname, "r")
        try:
            data = self.data_file["train_data"]
        except KeyError:
            data = self.data_file["data"]

        # extract validation data
        if "val_data" in self.data_file.keys():
            self.valid_fname = self.data_fname
        else:
            self.valid_fname = None

        # infer data dimensionalities
        self.N, self.D = data.shape

        # set data transform
        if "data_transform" in kwargs.keys():
            self.data_transform = kwargs["data_transform"]
        else:
            self.data_transform = None

    def extract_hypers_from_config(self, config):
        if self.is_hyperparameter_S:
            try:
                self.S = config["S"]
            except KeyError:
                raise KeyError(
                    "Number of states is not a hyperparameter, and none was passed " "to the init"
                )

        if self.is_hyperparameter_H:
            try:
                self.H = config["H"]
            except KeyError:
                raise KeyError(
                    "Number of initial hidden units is not a hyperparameter, and none "
                    "was passed to the init"
                )
