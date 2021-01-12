'''
Dataset and Dataloader classes for both numerical test cases 
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097
doi: https://dx.doi.org/10.3934/fods.2020019
github: https://github.com/zabaras/deep-turbulence
=====
'''
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.log import Log
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import random
import os

class TrainingDataset(Dataset):
    '''
    Training data-set with additive noise option for targets
    '''
    def __init__(self, inputs, targets, lstm_seeds, input_noise_std=0.01, tar_noise_std=0.0):
        """
        Training data-set for TM-Glow
        Args:   
        """
        assert inputs.size(0) == targets.size(0), 'inputs and target tensors must have same batch dimension size'
        assert inputs.size(0) == lstm_seeds.size(0), 'inputs and LSTM seed tensors must have same batch dimension size'
        self.inputs = inputs
        self.targets = targets
        self.lstm_seeds = lstm_seeds
        self.target_noise = tar_noise_std
        self.input_noise = input_noise_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input0 = self.inputs[idx] + self.input_noise*torch.randn(self.inputs[idx].size()).type(self.targets.type())
        target0 = self.targets[idx] + self.target_noise*torch.randn(self.targets[idx].size()).type(self.targets.type())
        lstm_seed0 = self.lstm_seeds[idx]
        return input0, target0, lstm_seed0

class TMGLowDataLoader(object):
    '''
    Parent class for TM-Glow dataloader creators
    Note: These are not used as actual data loaders, they create data loaders
    '''
    def __init__(self, training_dir='.', testing_dir='.', log=None):
        super().__init__()
        # Directory of data
        self.training_dir = training_dir
        self.testing_dir = testing_dir

        if(log is None):
            self.log = Log()
        else:
            self.log = log

        self.input_mean = None
        self.output_mean = None
        self.input_std = None
        self.output_std = None

    def readFluidData(self, input_file_name:str, target_file_name:str, fStride=1, cStride=1):

        coarse_file = os.path.join(self.training_dir, input_file_name)
        try:
            data_npz = np.load(coarse_file)
            self.log.log('Reading file {:s}.'.format(input_file_name), rec=False)
            # Remove z-velocity as it is not needed
            inputData = np.concatenate([data_npz['data'][::cStride,:2,:,:], data_npz['data'][::cStride,3:,:,:]], axis=1)
            # inputData.append(data_np)

            # inputTime.append(data_npz['times'])
        except FileNotFoundError:
            self.log.error("Uh-oh, seems a low-fidelity data file couldn't be found!")
            self.log.error('Check this file exists: {}'.format(coarse_file))
            inputData = None

        # Read in high-fidelity (target data)
        fine_file = os.path.join(self.training_dir, target_file_name)
        try:
            data_npz = np.load(fine_file)
            self.log.log('Reading file {:s}.'.format(target_file_name), rec=False)
            # Remove z-velocity as it is not needed
            targetData = np.concatenate([data_npz['data'][::fStride,:2,:,:], data_npz['data'][::fStride,3:,:,:]], axis=1)
            # targetData.append(data_np)

            # targetTime.append(data_npz['times'])
        except FileNotFoundError:
            self.log.error("Uh-oh, seems a high-fidelity data file couldn't be found!")
            self.log.error('Check this file exists: {}'.format(fine_file))
            targetData = None

        return inputData, targetData

    def calcNormalizingParams(self, inputData, targetData):
        '''
        Calculates the hyper-paramters used for normalizing the 
        training input/output data. Normalizes data to a standard unit Gaussian.
        Args:
            inputData (tensor): [b,t,c,d1,d2] tensor of low-fidelity inputs
            targetData (tensor): [b,t,c,d1*,d2*] tensor of high-fidelity target
        '''
        self.log.warning('Calculating normalizing constants')
        self.input_mean = torch.zeros(3)
        self.output_mean = torch.zeros(3)

        self.input_mean[0] = torch.mean(inputData[:,:,0])
        self.input_mean[1] = torch.mean(inputData[:,:,1])
        self.input_mean[2] = torch.mean(inputData[:,:,2])

        self.output_mean[0] = torch.mean(targetData[:,:,0])
        self.output_mean[1] = torch.mean(targetData[:,:,1])
        self.output_mean[2] = torch.mean(targetData[:,:,2])

        self.input_std = torch.zeros(3)+1
        self.output_std = torch.zeros(3)+1

        self.input_std[0] = torch.std(inputData[:,:,0])
        self.input_std[1] = torch.std(inputData[:,:,1])
        self.input_std[2] = torch.std(inputData[:,:,2])

        self.output_std[0] = torch.std(targetData[:,:,0])
        self.output_std[1] = torch.std(targetData[:,:,1])
        self.output_std[2] = torch.std(targetData[:,:,2])

    def setNormalizingParams(self, model):
        '''
        Given a PyTorch model this sets the normalizing paramters of
        the loader class using what is stored in the model. This is done
        to save normalizing constants between runs.
        Args:
            model: PyTorch model with normalizing constants as 
        '''
        self.input_mean = torch.zeros(3)
        self.output_mean = torch.zeros(3)
        self.input_mean = model.in_mu.cpu()
        self.output_mean = model.out_mu.cpu()

        self.input_std = torch.zeros(3)
        self.output_std = torch.zeros(3)
        self.input_std = model.in_std.cpu()
        self.output_std = model.out_std.cpu()

    def transferNormalizingParams(self, model):
        '''
        Given a PyTorch model this gets the calculated normalizing 
        parameters and assigned them to registered parameters of 
        the model. This is done to save normalizing constants between runs.
        Args:
            model: PyTorch model with normalizing constants params to be set
            device (PyTorch device): device the PyTorch model is on
        '''
        device = next(model.parameters()).device # Model's device
        model.in_mu = self.input_mean.to(device)
        model.out_mu = self.output_mean.to(device)

        model.in_std = self.input_std.to(device)
        model.out_std = self.output_std.to(device)

    def normalizeInputData(self, inputData):
        '''
        Normalize the input tensor on each channel (x-vel, y-vel, pressure) 
        '''
        # Normalize training data to unit Gaussian 
        inputData[:,:,0] = inputData[:,:,0] - self.input_mean[0]
        inputData[:,:,1] = inputData[:,:,1] - self.input_mean[1]
        inputData[:,:,2] = inputData[:,:,2] - self.input_mean[2]

        inputData[:,:,0] = inputData[:,:,0] / self.input_std[0]
        inputData[:,:,1] = inputData[:,:,1] / self.input_std[1]
        inputData[:,:,2] = inputData[:,:,2] / self.input_std[2]

        return inputData
    
    def normalizeTargetData(self, targetData):
        '''
        Normalize the target tensor on each channel (x-vel, y-vel, pressure)
        '''
        targetData[:,:,0] = targetData[:,:,0] - self.output_mean[0]
        targetData[:,:,1] = targetData[:,:,1] - self.output_mean[1]
        targetData[:,:,2] = targetData[:,:,2] - self.output_mean[2]

        targetData[:,:,0] = targetData[:,:,0] / self.output_std[0]
        targetData[:,:,1] = targetData[:,:,1] / self.output_std[1]
        targetData[:,:,2] = targetData[:,:,2] / self.output_std[2]

        return targetData

# /=================================================================================/
class BackwardStepLoader(TMGLowDataLoader):
    '''
    Class used for creating data loaders for the backwards step numerical example
    Args:
        ntrain (int): number of training data
        ntest (int): number of testing data
        data_dir (string): path of numpy data files
        shuffle (boolean): shuffle the training data or not
        log (Log): logging class
    '''
    def __init__(self, training_dir, testing_dir, shuffle=True, log=None):
        super().__init__(training_dir, testing_dir, log)
        self.shuffle = shuffle

    def createTrainingLoader(self, ntrain, u0, tSplit=1, inUpscale=1, batch_size=32, tar_noise_std=0):
        '''
        Creates the training loader
        Args:
            ntrain (np.array): Numpy array of training-indexes
            u0 (np.array): Input velocity for each training index (used for normalizing)
            tSplit (int): Number of time to split the simulation data into smaller time-series for training
            inUpscale (int): Initial upscaling, used to just make the architecture similar
            batch_size (int): Training batch-size
            tar_noise_std (int): Random noise to add on the target
        '''
        self.log.log('Creating backwards step training loader.')
        if(batch_size > len(ntrain*tSplit)):
            self.log.warning('Lowering mini-batch size to match training cases.')
            batch_size = len(ntrain*tSplit)

        inputData = []
        targetData = []
        u0Data = []
        # Loop through cases and read in each file
        for i, idx in enumerate(ntrain):
            inputData0, targetData0 = self.readFluidData("backwardStepCoarse{:d}-[U,p].npz".format(idx), "backwardStepFine{:d}-[U,p].npz".format(idx))
            inputData.append(inputData0)
            targetData.append(targetData0)
            u0Data.append(u0[idx])

        # Stack into single tensor.
        inputData = torch.Tensor(np.stack(inputData, axis=0))
        targetData = torch.Tensor(np.stack(targetData, axis=0))
        u0Data = torch.Tensor(u0Data)

        # Scale input if needed though interpolation
        inputData0 = []
        for i in range(inputData.size(1)):
            inputStep = F.interpolate(inputData[:,i], scale_factor=inUpscale, mode='bilinear', align_corners=True)
            inputData0.append(inputStep)
        inputData = torch.stack(inputData0, dim=1)
        
        # If normalizing parameters is not present, calculate them 
        u0 = u0Data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        u0 = torch.cat((u0, u0, u0**2), dim=2)
        
        inputData = inputData/u0
        targetData = targetData/u0

        u0 = u0Data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        u0 = u0.expand(u0.size(0), inputData.size(1), 1, inputData.size(-2), inputData.size(-1))
        inputData = torch.cat([inputData, u0], dim=2)

        if(self.input_mean is None or self.input_std is None):
            self.calcNormalizingParams(inputData, targetData)

        # Normalize the data by a unit Gaussian
        inputData = self.normalizeInputData(inputData)
        targetData = self.normalizeTargetData(targetData)

        # Split time-series into sub series
        self.log.log('Splitting time-series into {:d} chunks.'.format(tSplit))
        input_idx = int(inputData.size(1)//tSplit)
        target_idx = int(targetData.size(1)//tSplit)
        input_splits = []
        target_splits = []
        u0_splits = []

        for i in range(tSplit):
            rndIdx = np.random.randint(0, input_idx)
            input_splits.append(inputData[:,i*input_idx:(i+1)*input_idx])
            target_splits.append(targetData[:,i*target_idx:(i+1)*target_idx])
            u0_splits.append(u0Data)

        inputData = torch.cat(input_splits, dim=0)
        targetData = torch.cat(target_splits, dim=0)
        # Model is very sensistve to these initial states, thus they must
        # be the same between starting and stopping the training of the model
        c0Seeds = torch.LongTensor(inputData.size(0)).random_(0, 1000)

        dataset = TrainingDataset(inputData, targetData, c0Seeds, tar_noise_std)
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=False)

        return training_loader

    def createTestingLoader(self, ntest, u0, inUpscale=1, batch_size=32):
        '''
        Creates the testing loader
        Args:
            ntrain (np.array): Numpy array of training-indexes
            u0 (np.array): Input velocity for each training index (used for normalizing)
            inUpscale (int): Initial upscaling, used to just make the architecture similar
            batch_size (int): Training batch-size
        '''
        self.log.log('Creating backwards step testing loader.')
        if(batch_size > len(ntest)):
            self.log.warning('Lowering mini-batch size to match training cases.')
            batch_size = len(ntest)

        inputData = []
        targetData = []
        u0Data = []
        # Loop through cases and read in each file
        for i, idx in enumerate(ntest):
            inputData0, targetData0 = self.readFluidData("backwardStepCoarse{:d}-[U,p].npz".format(idx), "backwardStepFine{:d}-[U,p].npz".format(idx))
            inputData.append(inputData0)
            targetData.append(targetData0)
            u0Data.append(u0[idx])

        # Stack into single tensor.
        inputData = torch.Tensor(np.stack(inputData, axis=0))
        targetData = torch.Tensor(np.stack(targetData, axis=0))
        u0Data = torch.Tensor(u0Data)

        # Scale input if needed though interpolation
        inputData0 = []
        for i in range(inputData.size(1)):
            inputStep = F.interpolate(inputData[:,i], scale_factor=inUpscale, mode='bilinear', align_corners=True)
            inputData0.append(inputStep)
        inputData = torch.stack(inputData0, dim=1)
        
        # Normalize by the inlet velocity
        u0 = u0Data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        u0 = torch.cat((u0,u0,u0**2), dim=2)
        inputData = inputData/u0
        targetData = targetData/u0
        
        # Add inlet velocity as an input channel
        u0 = u0Data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        u0 = u0.expand(u0.size(0), inputData.size(1), 1, inputData.size(-2), inputData.size(-1))
        inputData = torch.cat([inputData, u0], dim=2)

        # If normalizing parameters is not present, calculate them 
        if(self.input_mean is None or self.input_std is None):
            self.calcNormalizingParams(inputData, targetData)

        # Normalize the data by a unit Gaussian
        inputData = self.normalizeInputData(inputData)
        targetData = self.normalizeTargetData(targetData)

        dataset = TensorDataset(inputData, targetData, u0Data)
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=False)

        return testing_loader

# /=================================================================================/
class CylinderArrayLoader(TMGLowDataLoader):
    '''
    Class used for creating data loaders for the cylinder array numerical example
    Args:
        ntrain (int): number of training data
        ntest (int): number of testing data
        data_dir (string): path of numpy data files
        shuffle (boolean): shuffle the training data or not
        log (Log): logging class
    '''
    def __init__(self, training_dir, testing_dir, shuffle=True, log=None,):
        super().__init__(training_dir, testing_dir, log)
        self.shuffle = shuffle

    def createTrainingLoader(self, ntrain, tSplit=1, inUpscale=1, batch_size=32, tar_noise_std=0):
        '''
        Creates the training loader
        Args:
            ntrain (np.array): Numpy array of training-indexes
            u0 (np.array): Input velocity for each training index (used for normalizing)
            tSplit (int): Number of time to split the simulation data into smaller time-series for training
            inUpscale (int): Initial upscaling, used to just make the architecture similar
            batch_size (int): Training batch-size
            tar_noise_std (int): Random noise to add on the target
        '''
        self.log.log('Creating cylinder array training loader.')
        if(batch_size > len(ntrain*tSplit)):
            self.log.warning('Lowering mini-batch size to match training cases.')
            batch_size = len(ntrain*tSplit)

        inputData = []
        targetData = []
        # Loop through cases and read in each file
        for i, idx in enumerate(ntrain):
            inputData0, targetData0 = self.readFluidData("cylinderArrayCoarse{:d}-[U,p].npz".format(idx), "cylinderArrayFine{:d}-[U,p].npz".format(idx))
            inputData.append(inputData0)
            targetData.append(targetData0)

        # Stack into single tensor.
        inputData = torch.Tensor(np.stack(inputData, axis=0))
        targetData = torch.Tensor(np.stack(targetData, axis=0))

        # Scale input if needed though interpolation
        inputData0 = []
        for i in range(inputData.size(1)):
            inputStep = F.interpolate(inputData[:,i], scale_factor=inUpscale, mode='bilinear', align_corners=True)
            inputData0.append(inputStep)
        inputData = torch.stack(inputData0, dim=1)
        
        if(self.input_mean is None or self.input_std is None):
            self.calcNormalizingParams(inputData, targetData)

        # Normalize the data by a unit Gaussian
        inputData = self.normalizeInputData(inputData)
        targetData = self.normalizeTargetData(targetData)

        # Split time-series into sub series
        self.log.log('Splitting time-series into {:d} chunks.'.format(tSplit))
        input_idx = int(inputData.size(1)//tSplit)
        target_idx = int(targetData.size(1)//tSplit)
        input_splits = []
        target_splits = []

        for i in range(tSplit):
           rndIdx = np.random.randint(0, input_idx)
           input_splits.append(inputData[:,i*input_idx:(i+1)*input_idx])
           target_splits.append(targetData[:,i*target_idx:(i+1)*target_idx])

        inputData = torch.cat(input_splits, dim=0)
        targetData = torch.cat(target_splits, dim=0)
        c0Seeds = torch.LongTensor(inputData.size(0)).random_(0, 1000)

        dataset = TrainingDataset(inputData, targetData, c0Seeds, tar_noise_std)
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=True)

        return training_loader

    def createTestingLoader(self, ntest, inUpscale=1, batch_size=32):
        '''
        Creates the testing loader
        Args:
            ntrain (np.array): Numpy array of training-indexes
            u0 (np.array): Input velocity for each training index (used for normalizing)
            inUpscale (int): Initial upscaling, used to just make the architecture similar
            batch_size (int): Training batch-size
        '''
        self.log.log('Creating cylinder array testing loader.')
        if(batch_size > len(ntest)):
            self.log.warning('Lowering mini-batch size to match training cases.')
            batch_size = len(ntest)

        inputData = []
        targetData = []
        # Loop through cases and read in each file
        for i, idx in enumerate(ntest):
            inputData0, targetData0 = self.readFluidData("cylinderArrayCoarse{:d}-[U,p].npz".format(idx), "cylinderArrayFine{:d}-[U,p].npz".format(idx))
            inputData.append(inputData0)
            targetData.append(targetData0)

        # Stack into single tensor.
        inputData = torch.Tensor(np.stack(inputData, axis=0))
        targetData = torch.Tensor(np.stack(targetData, axis=0))

        # Scale input if needed though interpolation
        inputData0 = []
        for i in range(inputData.size(1)):
            inputStep = F.interpolate(inputData[:,i], scale_factor=inUpscale, mode='bilinear', align_corners=True)
            inputData0.append(inputStep)
        inputData = torch.stack(inputData0, dim=1)

        if(self.input_mean is None or self.input_std is None):
            self.calcNormalizingParams(inputData, targetData)

        # Normalize the data by a unit Gaussian
        inputData = self.normalizeInputData(inputData)
        targetData = self.normalizeTargetData(targetData)
        u0Data = torch.ones(inputData.size(0))

        dataset = TensorDataset(inputData, targetData, u0Data)
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=False)

        return testing_loader
    

class DataLoaderAuto(object):
    @classmethod
    def init_data_loaders(cls, args, model, log):
        '''
        Used to initialize data-loaders for each numerical test case
        Args:

        '''
         # First check if the model name is a pre-defined config
        if(args.exp_type == 'backward-step'):
            return cls.setupBackwardStepLoaders(args, model, log)
        elif(args.exp_type == 'cylinder-array'):
            return cls.setupCylinderLoaders(args, model, log)
        else:
            raise AssertionError("Provided experiment name, {:s}, not supported .".format(args.exp_type))

    
    @classmethod
    def setupBackwardStepLoaders(cls, args, model, log):
        # Set up cylinder training and testing loaders
        log.log('Setting up backward step loaders.')
        ntrain = np.arange(0,64,1)
        np.random.seed(args.seed)
        np.random.shuffle(ntrain)
        ntest = ntrain[-args.ntest:]
        ntrain = np.linspace(0,63,args.ntrain).astype(int)
        
        u0 = np.linspace(1,10,64)
        stepLoader = BackwardStepLoader(args.training_data_dir, args.testing_data_dir)

        if(args.epoch_start > 0):
            # If starting from an epoch load data normalization constants from the model's buffers
            stepLoader.setNormalizingParams(model.module)
            training_loader = stepLoader.createTrainingLoader(ntrain, u0, tSplit=2, \
                inUpscale=(1.34,1.34), batch_size=args.batch_size, tar_noise_std=args.noise_std)
        else:
            training_loader = stepLoader.createTrainingLoader(ntrain, u0, tSplit=2, \
                inUpscale=(1.34,1.34), batch_size=args.batch_size, tar_noise_std=args.noise_std)
            stepLoader.transferNormalizingParams(model.module)
            
        testing_loader = stepLoader.createTestingLoader(ntest, u0, inUpscale=(1.34,1.34), batch_size=args.test_batch_size)

        return stepLoader, training_loader, testing_loader

    @classmethod
    def setupCylinderLoaders(cls, args, model, log):
        # Set up cylinder training and testing loaders
        log.log('Setting up cylinder array loaders.')
        ntest = np.arange(96,96+args.ntest,1).astype(int)
        ntrain = np.linspace(0,95,args.ntrain).astype(int)
        
        cylinder_loader = CylinderArrayLoader(args.training_data_dir, args.testing_data_dir)

        if(args.epoch_start > 0):
            # If starting from an epoch load data normalization constants from the model's buffers
            cylinder_loader.setNormalizingParams(model.module)
            training_loader = cylinder_loader.createTrainingLoader(ntrain, tSplit=2, batch_size=args.batch_size)
        else:
            training_loader = cylinder_loader.createTrainingLoader(ntrain, tSplit=2, batch_size=args.batch_size)
            cylinder_loader.transferNormalizingParams(model.module)
            
        testing_loader = cylinder_loader.createTestingLoader(ntest, batch_size=args.test_batch_size)
        return cylinder_loader, training_loader, testing_loader
