'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: https://github.com/zabaras/deep-turbulence
=====
'''
from utils.utils import toNumpy, toTuple, getGpuMemoryMap
from utils.viz import plotVelocityPred, plotNumericalPred
from utils.log import Log
from utils.parallel import DataParallelCriterion
from torch import autograd
from pc.physicsConstrained import PhysConstrainedLES

import torch
from torch import nn
from typing import Optional, TypeVar
from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F
import os
import math
import copy

tensorList = TypeVar('tensorList', torch.Tensor, list)  # Must be str or bytes
@dataclass
class TMGLowPredictionItem:
    '''
    Class used to store model predictions overtime
    '''
    yPred: Optional[tensorList] = None
    yTarget: Optional[tensorList] = None
    logp: Optional[tensorList] = None
    tback: int = 1

    def add(self, yPred0, logp0, yTarget0):
        '''
        Adds provided prediction and target tensors to class
        Time-steps are stored in dimension 1
        '''
        yPred0 = self.unsqueeze(yPred0, dim=1)
        logp0 = self.unsqueeze(logp0, dim=1)
        yTarget0 = self.unsqueeze(yTarget0, dim=1)
        if(self.yPred is None or self.yTarget is None):
            self.yPred = yPred0
            self.logp = logp0
            self.yTarget = yTarget0
        else:
            self.yPred = self.concat(self.yPred, yPred0)
            self.logp = self.concat(self.logp, logp0)
            self.yTarget = self.concat(self.yTarget, yTarget0)

    def getOutputs(self):
        '''
        Combines yPred and logp into a single tuple for loss evaluation
        '''
        if(isinstance(self.yPred, list)):
            return [(self.yPred[i], self.logp[i]) for i in range(len(self.yPred))]
        else:
            return (self.yPred, self.logp)
    
    def getTargets(self, *newTargets):
        '''
        Returns target tensors and allows for additional target tensors
        to be added into a tuple
        '''
        if(isinstance(newTargets, tuple) and len(newTargets) > 0):
            print(type(newTargets))
            return (self.yTarget,) + newTargets
        else:
            return self.yTarget

    def unsqueeze(self, tensor, dim=1):
        '''
        Unsqueezes the tensor or list of tensors in the specified dimension
        '''
        if(isinstance(tensor, list)):
            return [tensor[i].unsqueeze(dim) for i in range(len(tensor))]
        else:
            return tensor.unsqueeze(dim)

    def concat(self, tensor1, tensor2, dim=1):
        '''
        Concats the new tensor 2 onto existing series tensor 1
        '''
        assert type(tensor1) == type(tensor2), "Tensor 1 of type {} is not the same as Tensor 2 with type {}".format(type(tensor1),type(tensor2))
        if(isinstance(tensor1, list)):
            assert len(tensor1) == len(tensor2), "List sizes of tensors are not equal."
            return [torch.cat([tensor1[i], tensor2[i]], dim=dim) for i in range(len(tensor2))]
        else:
            return torch.cat([tensor1, tensor2], dim=dim)
    
    def clear(self):
        '''
        Clears stored prediction and target variables. Should be called after backprop
        '''
        self.yPred = None
        self.logp = None
        self.yTarget = None
        

class TMGLowLoss(nn.Module):
    '''
    A torch.nn module for executing loss calculations on multiple GPUs.
    Args:
        args (argparse): object with programs arguments
        model (torch.nn): Glow-TM model
    '''
    def __init__(self, args, model, log=None):
        super(TMGLowLoss, self).__init__()

        self.beta = args.beta
        self.phys = PhysConstrainedLES(args.dx, args.dy, grad_kernels=[3, 3])
        # Normalizing paramters needed to un-normalize the predictions for PDE
        # residual calculations.
        self.register_buffer("output_std", model.module.out_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer("output_mu", model.module.out_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    def forward(self, yPred, logp, target, target_mean, target_rms):
        '''
        The forward of this class is called through the parallel wrapper
        Thus this supports the extention of the loss calculation to different GPUs
        Args:
            yPred (torch.Tensor): [batch x time-steps x 3 x nx x ny] Glow-TM prediction
            logp (torch.Tensor): The log probability of the likelihood (entropy term)
            yPred (torch.Tensor): [batch x time-steps x 3 x nx x ny] Glow-TM prediction
            target_mean (torch.Tensor): [batch x 3 x nx x ny] mean flow field to use in RMS predictions
            target_rms (torch.Tensor): [batch x 3 x nx x ny] target RMS value of flucation states
        '''
        vPres = self.calcVPres(yPred.view(-1,yPred.size(-3),yPred.size(-2),yPred.size(-1)))

        vDiv = self.calcVDiv(yPred.view(-1,yPred.size(-3),yPred.size(-2),yPred.size(-1)))

        yPredHat = self.output_std*yPred + self.output_mu
        targetHat = self.output_std*target + self.output_mu
        targetMeanHat = self.output_std*target_mean.unsqueeze(1) + self.output_mu

        vL1 = torch.mean(torch.pow(yPred - target, 2))

        pred_rms = torch.sqrt(torch.mean((yPred - torch.mean(yPred, dim=1).unsqueeze(1))**2, dim=1))
        vRMS = torch.mean(torch.pow(pred_rms - target_rms, 2))

        # Calc the negative entropy
        n_out_pixels = yPred.size(-3)*yPred.size(-2)*yPred.size(-1)
        neg_entropy = logp.mean() / math.log(2.) / n_out_pixels # Divide by log 2 here since entropy unit is bit
        # Calc. Reverse KL-Loss
        loss_step = (self.beta*(vPres + vDiv + vL1 + vRMS) + neg_entropy)

        return loss_step

    def calcVPres(self, yPred):
        '''
        Calculates the pressure residual term.
        Args:
            yPred (Tensor): [b x 3 x d1 x d2]
        '''
        # First un-normalize output
        yPredHat = self.output_std*yPred + self.output_mu
        # Calculate the pressure poisson residual
        pStar = self.phys.calcPressurePoisson(yPredHat[:,:2], yPredHat[:,2:])
        # L2 Norm of residual
        return torch.mean(torch.pow(pStar[:,:,1:-1,1:-1], 2))

    def calcVDiv(self, yPred):
        '''
        Calculates the divergence residual term.
        Args:
            yPred (Tensor): [b x 3 x d1 x d2]
        '''
        # First un-normalize output
        yPredHat = self.output_std*yPred + self.output_mu
        # Calculate the divergence residual
        uStar = self.phys.calcDivergence(yPredHat[:,:2])
        # L2 Norm of residual
        return torch.mean(torch.pow(uStar[:,:,1:-1,1:-1],2))


class TrainFlow(object):
    '''
    Trains recursive encoder attention-driven decoder
    Args:
        args (argparse): object with programs arguments
        model (torch.nn): Glow-TM model
        train_loader (torch.dataloader): dataloader with training cases
        test_loader (torch.dataloader): dataloader with training cases
        log (Log): class for logging console outputs
    '''
    def __init__(self, args, model, train_loader, test_loader, log=None):
        super().__init__()
        self.args = args
        self.trainingLoader = train_loader
        self.testingLoader = test_loader
        
        if(log is None):
            self.log = Log(self.args)
        else:
            self.log = log

        loss = TMGLowLoss(args, model).to(args.src_device)
        self.parallel_loss  = DataParallelCriterion(loss, args.device_ids)
    
    def trainParallel(self, model, optimizer, tback=1, epoch=0, **kwargs):
        '''
        Trains the model for a single epoch
        Args:
            model (torch.nn.Module): PyTorch model to train
            optimizer (torch.optim): PyTorch optimizer to update the models parameters
            tback (int): number of time-steps to back propagate through in time
            stride (int): The stride the low-fidelity input takes compared to output
            epoch (int): current epoch
        Returns:
            total_loss (float): current loss
        '''
        model.train()
        # Total training loss
        total_loss = 0
        beta = self.args.beta
        
        print("Beta:", beta)
        optimizer.zero_grad()
        for mbIdx, (input0, target0, lstm_seeds) in enumerate(self.trainingLoader):

            aKey = model.module.initLSTMStates(lstm_seeds, [target0.size(-2), target0.size(-1)])
            # aKey = model.module.initLSTMStates(torch.LongTensor(input0.size(0)).random_(0, int(1e8)), [target0.size(-2), target0.size(-1)])
            a0 = copy.deepcopy(aKey)

            loss = 0 # Time-series loss
            # Loop of time-steps
            tmax = target0.size(1)    
            tback = 10
            
            input_next = input0[:,:tback].to(self.args.device)
            target_next = target0[:,:tback].to(self.args.device)
            
            target0_mean = torch.mean(target0, axis=1).to(self.args.device)
            target0_rms = torch.sqrt(torch.mean((target0.to(self.args.device) - target0_mean.unsqueeze(1))**2, dim=1)).to(self.args.device)

            # Splits time-series into smaller blocks to calculate back-prop through time
            for i in range(0, tmax//tback):

                input = input_next
                ytarget = target_next

                # Asynch load the next time-series
                if(i+1 < tmax//tback):
                    input_next = input0[:,(i+1)*tback:(i+2)*tback].cuda(self.args.device, non_blocking=True)
                    target_next = target0[:,(i+1)*tback:(i+2)*tback].cuda(self.args.device, non_blocking=True)

                loss = 0 
                gpu_loss = [0 for i in range(self.args.n_gpu)]
                modelPredictions = TMGLowPredictionItem()
                model.scatterModel()
                
                for tstep in range(tback):

                    # Model forward
                    outputs = model.sample(input[:,tstep], a0)
                    
                    if(isinstance(outputs, list)):
                        yPred = [output[0] for output in outputs]
                        logp = [output[1] for output in outputs]
                        a0 = [output[2] for output in outputs]      
                    else:
                        yPred, logp, a0 = outputs

                    modelPredictions.add(yPred, logp, ytarget[:,tstep])
                    # Recompile recurrent states onto the source device
                    if(self.args.n_gpu > 1):
                        a0 = model.gatherLSTMStates(a0)
                    else:
                        a0 = outputs[2]

                # Compute the reverse KL divergence loss
                outputs = modelPredictions.getOutputs()
                targets = modelPredictions.getTargets()
                loss0 = self.parallel_loss(outputs, targets, target0_mean, target0_rms)
                if(self.args.n_gpu > 1):
                    gpu_loss = [gpu_loss[i] + loss0[i] for i in range(len(loss0))]
                else:
                    gpu_loss = [gpu_loss[0] + loss0]

                modelPredictions.clear()
                loss = self.parallel_loss.gather(gpu_loss, output_device=self.args.src_device).mean()
                # Backwards!
                loss.backward()

                # print(getGpuMemoryMap())
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                # Average the LSTM states with the initial state to prevent convergence
                for j in range(len(a0)):
                    a_out, c_out = a0[j]
                    a_key, c_key = aKey[j]
                    a0[j] = (0.5*a_out.detach() + 0.5*a_key, 0.5*c_out.detach() + 0.5*c_key)

                total_loss = total_loss + loss.detach()
                # Sync cuda processes here
                # Note sure if needed, but hopefully makes sure next data is loaded.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Add loss of time-series to total loss   
            # Mini-batch progress log
            if ((mbIdx+1) % 5 == 0):
                self.log.log('Train Epoch: {}; Mini-batch: {}/{} ({:.0f}%); \t Current Loss: {:.6f}'.format(
                epoch, mbIdx, len(self.trainingLoader), 100. * mbIdx / len(self.trainingLoader), total_loss))

        return total_loss

    def test(self, model, samples=1, epoch=0, plot=True):
        '''
        Tests the model
        Args:
            model (torch.nn.Module): PyTorch model to test
            stride (int): The stride the low-fidelity input takes compared to output
            samples (int): Number of prediction to sample from the model
            epoch (int): current epoch
            plot (boolean): If to plot two of the predictions or not
        Returns:
            mse (float): mean-squared-error between the predictive mean and targets
        '''
        model.eval()
        # Total test loss
        total_loss = 0
        out_std = model.module.out_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out_mu = model.module.out_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        for mbIdx, (input0, target0, u0) in enumerate(self.testingLoader):

            u0 = u0.to(self.args.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            u0 = torch.cat((u0/2.0,u0/2.0,u0**2), dim=2)

            u0in = torch.ones(input0[:,:,:1].size()).to(self.args.device)*u0[:,:,0,:,:].unsqueeze(2)
            # input = torch.cat((input0.to(self.args.device), u0in), dim=2)
            input = input0.to(self.args.device)

            ytarget = out_std*target0.to(self.args.device) + out_mu

            dims = [samples]+list(ytarget.size())
            yPred = torch.zeros(dims).type(ytarget.type())
            
            # Max number of time steps
            tmax = 40   
            # Loop through samples
            for i in range(samples):
                
                aKey =model.module.initLSTMStates(torch.LongTensor(input.size(0)).random_(0, int(1e8)), [ytarget.size(-2), ytarget.size(-1)])
                a0 = copy.deepcopy(aKey)

                # Loop of time-steps
                model.scatterModel()
                
                for tstep in range(0, tmax+1):

                    # Model forward
                    outputs = model.sample(input[:,tstep], a0)
                    yPred0, logp, a0 = model.gather(outputs, self.args.src_device)

                    out_std = model.module.out_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    out_mu = model.module.out_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    yPredHat = out_std*yPred0 + out_mu
                    yPred[i,:,tstep] = yPredHat.detach()

                    # Average current LSTM states with initial state
                    if(tstep%10 == 0):
                        for j in range(len(a0)):
                            a_out, c_out = a0[j]
                            a_key, c_key = aKey[j]
                            a0[j] = (0.5*a_out.detach() + 0.5*a_key, 0.5*c_out.detach() + 0.5*c_key)

            if(plot and mbIdx == 0):
                self.log.log('Plotting predictions.')
                plotVelocityPred(self.args, input, yPred, ytarget, bidx=0, stride=4, epoch=epoch)
                plotVelocityPred(self.args, input, yPred, ytarget, bidx=1, stride=4, epoch=epoch)

            # Summation of the squared error between the mean of the samples and target
            total_loss = total_loss + (torch.pow(torch.mean(yPred[:,:,1:tmax+1], dim=0) - ytarget[:,1:tmax+1], 2)).sum().detach()

        # Return the mse
        return total_loss/(self.args.ntest*tmax*yPred.size(-2)*yPred.size(-1))
