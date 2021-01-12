'''
The main run file for training TM-Glow for both the backwards step 
and cylinder array test cases which can be controlled through the
arguments.
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097
doi: https://dx.doi.org/10.3934/fods.2020019
github: https://github.com/zabaras/deep-turbulence
=====
'''
from args import Parser
from nn.tmGlow import TMGlow
from nn.trainFlowParallel import TrainFlow

from utils.dataLoader import DataLoaderAuto
from utils.utils import saveWorkspace, loadWorkspace
from utils.log import Log
from utils.parallel import DataParallelINNModel

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, CyclicLR

import torch
import numpy as np
import os, sys

if __name__ == '__main__':

    # Parse arguments
    args = Parser().parse()    
    if(args.epoch_start > 0):
        print('Looking to load workspace {:d}.'.format(args.epoch_start))
        args, model_state_dict, optimizer_state_dict = loadWorkspace(args, args.ckpt_dir, file_id=args.epoch_start)
    log = Log(args, record=True)

    # Set up PyTorch devices
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Torch device:{}".format(args.device))
    if(torch.cuda.device_count() > 1 and args.parallel == True):
        if(torch.cuda.device_count() < args.n_gpu):
            args.n_gpu = torch.cuda.device_count()
        if(args.n_gpu < 1):
            args.n_gpu = torch.cuda.device_count()
        log.info("Looks like we have {:d} GPUs to use. Going parallel.".format(args.n_gpu))
        args.device_ids = [i for i in range(0,args.n_gpu)]
        args.src_device = "cuda:{}".format(args.device_ids[0])
    else:
        log.info("Using a single GPU for training.")
        args.device_ids = [0]
        args.src_device = "cuda:{}".format(args.device_ids[0])
        args.parallel == False
        args.n_gpu = 1

    # Construct the model
    scheduler = None
    
    # def __init__(self, in_channels, out_channels, nsteps, enc_blocks, glow_blocks, 
    #             cond_features=8, cglow_upscale=1, growth_rate=4, init_features=48, rec_features=8, bn_size=8, 
    #             drop_rate=0, bottleneck=False)
    model = TMGlow(in_features=args.nic, 
                out_features=args.noc,
                enc_blocks=args.enc_blocks,
                glow_blocks=args.glow_blocks,
                cond_features=args.cond_features,
                cglow_upscale=args.glow_upscale, 
                growth_rate=args.growth_rate, 
                init_features=args.init_features,
                rec_features=args.rec_features,
                bn_size=args.bn_size,
                drop_rate=args.drop_rate,
                bottleneck=False).to(args.src_device)

    # Wrap model module with parallel GPU support
    # Can also handle a single GPU as well
    model = DataParallelINNModel(model, args.device_ids)
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*(0.995**args.epoch_start), weight_decay=1e-8, amsgrad=True)
    scheduler = ExponentialLR(optimizer, gamma=0.995)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 13, 2)

    if(args.epoch_start > 0):
         model.module.load_state_dict(model_state_dict)
         optimizer.load_state_dict(optimizer_state_dict)

    data_loader, training_loader, testing_loader = DataLoaderAuto.init_data_loaders(args, model, log)
    modelTrainer = TrainFlow(args, model, training_loader, testing_loader, log=log)

    # ========== Epoch loop ============
    log.log('Training network, lets rock.')
    for epoch in range(args.epoch_start+1, args.epochs + 1):

        # Time-step size to take
        tstep = (int(epoch/2)+10)
        # tstep = (int(epoch/10)+2)
        log.log('tsteps: {}'.format(tstep))

        loss = modelTrainer.trainParallel(model, optimizer, epoch=epoch)
        log.log('Epoch {:d}: Sample Training Loss: {}'.format(epoch, loss))
        
        if(not scheduler is None):
            scheduler.step()
            for param_group in optimizer.param_groups:
                log.log('Learning-rate: {:0.05f}'.format(param_group['lr']))

        if(epoch % args.test_freq == 0):
            log.log('Testing Model')
            with torch.no_grad():
                loss = modelTrainer.test(model, samples=2, epoch=epoch)
            log.log('Epoch {:d}: Testing Loss: {}'.format(epoch, loss))

        if(epoch % args.ckpt_freq == 0):
            file_dir = args.ckpt_dir
            # If director does not exist create it
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            log.log('Epoch {}, Saving network!'.format(epoch))
            # Note, we save the base model created on the source device
            saveWorkspace(args, model.module, optimizer, file_id=epoch)

