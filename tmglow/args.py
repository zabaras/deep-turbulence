'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/zabaras/deep-turbulence
=====
'''
from pprint import pprint
from dataclasses import dataclass
from collections import OrderedDict
import argparse
import time
import json
import torch
import random
import os
import errno

@dataclass
class BackwardStepConfig:
    """Configuration Dataclass to setup the model for the backward-step numerical test case."""
    parallel:bool = True
    n_gpu:int = 4
    
    # Data parameters
    ntrain: int = 48
    ntest: int = 4
    noise_std:float = 0.05
    training_data_dir:str = '../step-training/'
    testing_data_dir:str = '../step-testing/'

    # Training/Loss parameters
    beta:float = 200
    dx:float = 2./64
    dy:float = 2./64
    nu:float = 0.005
    batch_size:int = 32
    test_batch_size:int = 4
    max_grad_norm:float = 0.01

    # Model related
    nic: int =4
    glow_upscale:int = 2 # Upscale by two because we will upscale low-fidelity input prior

@dataclass
class CylinderConfig:
    """Configuration Dataclass to setup the model for the cylinder-array numerical test case."""
    parallel:bool = True
    n_gpu:int = 4
    
    # Data parameters
    ntrain: int = 96
    ntest: int = 4
    noise_std:float = 0.05
    training_data_dir:str = '../cylinder-training/'
    testing_data_dir:str = '../cylinder-testing/'

    # Training/Loss parameters
    beta:float = 200
    dx:float = 5./64
    dy:float = 5./64
    nu:float = 0.005
    batch_size:int = 64
    test_batch_size:int = 4
    max_grad_norm:float = 1.0

    # Model related
    nic: int = 3
    glow_upscale:int = 4

CONFIG_MAPPING = OrderedDict(
    [
        ("backward-step", BackwardStepConfig),
        ("cylinder-array", CylinderConfig),
    ]
)

class Parser(argparse.ArgumentParser):
    """Program arguments, only a few are listed in the documentation.

    :param exp-dir: Directory to save experiments
    :type exp-dir: string
    :param exp-type: Experiment type
    :type exp-type: string
    :param parallel: Use parallel GPUs for training
    :type parallel: bool
    :param n_gpu: Number of GPUs to use for training, defaults to 1
    :type n_gpu: int
    :param training_data_dir: File directory to training data
    :type training_data_dir: string
    :param testing_data_dir: File directory to testing data
    :type testing_data_dir: string
    :param ntrain: Number of training data
    :type ntrain: int
    :param ntest: Number of testing data
    :type ntest: int
    :param epoch_start: Epoch to start at, will load pre-trained network
    :type epoch_start: int
    :param epochs: Number of epochs to train
    :type epochs: int
    :param lr: ADAM optimizer learning rate
    :type lr: float

    :note: Use `python main.py --help` for more information. Only several key of arguments are listed here.
    """
    def __init__(self):
        super(Parser, self).__init__(description='Read')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')
        self.add_argument('--exp-type', type=str, default='backward-step', choices=['backward-step', 'cylinder-array'], help='experiment')
        self.add_argument('--parallel', type=bool, default=0, help='use parallel GPUs for training')
        self.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use for training')   
        
        # model      
        self.add_argument('--enc-blocks', nargs="*", type=int, default=[4,4,4], help='list of encoder blocks')
        self.add_argument('--glow-blocks', nargs="*", type=int, default=[16,16,16], help='list of conditional flow blocks')
        self.add_argument('--growth-rate', type=int, default=4, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=16, help='number of initial features after the first conv block')
        self.add_argument('--rec-features', type=int, default=64, help='number of recurrent features used in the lstm glow')
        self.add_argument('--cond-features', type=int, default=32, help='number of conditional features pasted to glow from the encoder')
        self.add_argument('--drop-rate', type=float, default=0.0, help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enables bottleneck design in the dense blocks')

        # data
        self.add_argument('--training_data_dir', type=str, default=None, help="file directory to training data")
        self.add_argument('--testing_data_dir', type=str, default=None, help="file directory to testing data")
        self.add_argument('--ntrain', type=int, default=None, help="number of training data")
        self.add_argument('--ntest', type=int, default=None, help="number of test data")
        self.add_argument('--noise-std', type=float, default=None, help='relative noise std')
        self.add_argument('--max-grad-norm', type=float, default=0.1, help='gradient clipping value')

        # more details on dataset
        self.add_argument('--noc', type=int, default=3, help="number of output channels")
        self.add_argument('--nic', type=int, default=None, help="number of input channels per time-step")
        self.add_argument('--glow_upscale', type=int, default=None, help="The upscale ratio of the high-fidelity output to the low-fidelity input")
        self.add_argument('--nsteps', type=int, default=10, help="number of input time steps to use")
        self.add_argument('--nel', type=int, default=64, help="number of elements/ collocation points")

        # training
        self.add_argument('--epoch_start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--beta', type=float, default=None, help='temperature parameter in Boltzmann distribution')
        self.add_argument('--dx', type=float, default=None,  help='mesh discretization in the x direction')
        self.add_argument('--dy', type=float, default=None,  help='mesh discretization in the y direction')
        self.add_argument('--nu', type=float, default=None,  help='viscosity of fluid system')
        self.add_argument('--batch-size', type=int, default=None, help='batch size for training')
        self.add_argument('--test-batch-size', type=int, default=None, help='batch size for testing')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        
        # logging
        self.add_argument('--plot-freq', type=int, default=25, help='how many epochs to wait before plotting test output')
        self.add_argument('--test-freq', type=int, default=5, help='how many epochs to test the model')
        self.add_argument('--ckpt-freq', type=int, default=5, help='how many epochs to wait before saving model')
        self.add_argument('--notes', type=str, default="")

    def parse(self, dirs=True):
        '''
        Parse program arguments
        Args:
        dirs (boolean): True to make file directories for predictions and models
        '''
        args = self.loadConfig(self.parse_args())
        # Experiment run directory
        if len(args.notes) > 0:
            args.run_dir = args.exp_dir + '/' + '{}'.format(args.exp_type) \
                + '/ntrain{}_batch{}_blcks{}_lr{}_{}'.format(args.ntrain, args.batch_size, args.enc_blocks, args.lr, args.notes)
        else:
            args.run_dir = args.exp_dir + '/' + '{}'.format(args.exp_type) \
                + '/ntrain{}_batch{}_blcks{}_lr{}'.format(args.ntrain, args.batch_size, args.enc_blocks, args.lr)

        args.ckpt_dir = args.run_dir + '/checkpoints'
        args.pred_dir = args.run_dir + "/predictions"
        if(dirs):
            self.mkdirs(args.run_dir, args.ckpt_dir, args.pred_dir)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # print('Arguments:')
        # pprint(vars(args))

        if dirs:
            with open(args.run_dir + "/args.json", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

    def loadConfig(self, args):
        '''
        Loads experimental configurations.
        '''
        # First check if the model name is a pre-defined config
        if(args.exp_type in CONFIG_MAPPING.keys()):
            config_class = CONFIG_MAPPING[args.exp_type]
            # Init config class
            config = config_class()
            for attr, value in config.__dict__.items():
                if not hasattr(args, attr) or getattr(args, attr) is None:
                    setattr(args, attr, value)
        else:
            raise AssertionError("Provided experiment name, {:s}, not found in experiment list.".format(args.exp_type))

        return args

    def mkdirs(self, *directories):
        '''
        Makes a directory if it does not exist
        '''
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
