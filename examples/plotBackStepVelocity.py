'''
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097
doi: https://dx.doi.org/10.3934/fods.2020019
github: https://github.com/zabaras/deep-turbulence
=====
'''
import sys
sys.path.append('../..')

import torch
import numpy as np
import os, sys

from tmglow.args import Parser
from tmglow.nn.tmGlow import LSTMCGlow
from tmglow.utils.dataLoader import BackwardStepLoader
from tmglow.utils.utils import saveWorkspace, loadWorkspace, modelPred
from tmglow.utils.log import Log

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import copy
import time

def createColorBarHorizontal(fig, ax0, c_min, c_max, label_format = "{:02.2f}", cmap='viridis'):
    
    p0 = ax0.get_position().get_points().flatten()
    # ax_cbar = fig.add_axes([p0[2]+0.005, p0[1], 0.020, p0[3]-p0[1]])
    ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = [label_format.format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='horizontal', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

def createColorBarVertical(fig, ax0, c_min, c_max, label_format = "{:02.2f}", cmap='viridis'):
    
    p0 = ax0[0,-1].get_position().get_points().flatten()
    p1 = ax0[-2,-1].get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p1[2]+0.005, p1[1], 0.0075, p0[3]-p1[1]])
    # ax_cbar = fig.add_axes([p0[0], p0[1]-0.075, p0[2]-p0[0], 0.02])
    ticks = np.linspace(0, 1, 5)
    tickLabels = np.linspace(c_min, c_max, 5)
    tickLabels = [label_format.format(t0) for t0 in tickLabels]
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=plt.get_cmap(cmap), orientation='vertical', ticks=ticks)
    cbar.set_ticklabels(tickLabels)

def plotPredictionSeries(input0, yPred0, yTarget0, bidx=0, nsteps=10, stride=1, nsamp=1):

    input0 = input0.detach().cpu().numpy()
    yPred0 = yPred0.detach().cpu().numpy()
    yTarget0 = yTarget0.detach().cpu().numpy()

    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    # mpl.rcParams['figure.dpi'] = 300
    rc('text', usetex=True)

    cmap0 = 'Spectral_r'
    cmap0_var = 'inferno'
    fields = ['ux', 'uy', 'p']

    # for cidx in range(len(fields)):

    fig, ax = plt.subplots(3+nsamp, nsteps, figsize=(3.5*nsteps, nsamp+2))

    tmax = nsteps*stride
    input_mag = np.sqrt(input0[bidx,:,0]**2 + input0[bidx,:,1]**2)
    target_mag = np.sqrt(yTarget0[bidx,:,0]**2 + yTarget0[bidx,:,1]**2)
    pred_mag = np.sqrt(yPred0[:,bidx,:,0]**2 + yPred0[:,bidx,:,1]**2)
    pred_mag_std = np.std(pred_mag, axis=0)

    c_max = max([np.amax(input_mag), np.amax(target_mag)])

    times = np.arange(0,40,0.5)
    for t0 in range(nsteps):
        # High fidelity data
        ax[0, t0].imshow(target_mag[stride*(t0+1)], interpolation='nearest',  extent=[0,10,-1,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=0)
        # Low-fidelity data
        ax[1, t0].imshow(input_mag[stride*(t0+1)], interpolation='nearest',  extent=[0,10,-1,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=0)
        
        # Mean prediction
        for i in range(nsamp):
            pcm = ax[2+i, t0].imshow(pred_mag[i,stride*(t0+1)], interpolation='nearest',  extent=[0,10,-1,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=0)

        # Standard deviation
        pcm = ax[-1, t0].imshow(pred_mag_std[stride*(t0+1)], interpolation='nearest',  extent=[0,10,-1,1], cmap=cmap0_var, origin='lower', vmin=0)
        createColorBarHorizontal(fig, ax[-1, t0], 0, np.amax(pred_mag_std[stride*(t0+1)]), cmap=cmap0_var) # Variance cbar

        ax[0, t0].set_title('t={:.01f}'.format(times[stride*(t0+1)]), fontsize=18)
        # Set ticks and labels
        for j in range(3+nsamp):
            ax[j, t0].set_yticks([-1,0,1])
            if(t0 > 0):
                ax[j, t0].set_yticklabels([])
            else:
                ax[j, t0].set_yticklabels([-1, 0, 1])
            ax[j, t0].set_xticks([0,2,4,6,8,10])
            ax[j, t0].set_xticklabels([])
        ax[-1, t0].set_xticklabels([0,2,4,6,8,10])

    createColorBarVertical(fig, ax, 0, c_max, cmap=cmap0) # Samples cbar

    file_dir = './imgs'
    # If director does not exist create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    # Gif
    file_name = file_dir+"/predMagSampleSeries{:d}".format(bidx)
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')

if __name__ == '__main__':

    # Parse arguements
    args = Parser().parse(dirs=False)    
    
    use_cuda = "cpu"
    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device:{}".format(args.device))

    log = Log(args, record=False)

    # Load pretrained model parameters
    model_name = 'bstepWorkspace'
    model_id = 400
    model_path = '.'
    args, model_state_dict, optimizer_state_dict = loadWorkspace(args, model_path, file_name=model_name, file_id=model_id)
    scheduler = None

    model = LSTMCGlow(in_channels=args.nic, 
            out_channels=args.noc,
            nsteps=1,
            enc_blocks=args.enc_blocks,
            glow_blocks=args.glow_blocks,
            cond_features=args.cond_features,
            cglow_upscale=2, 
            growth_rate=args.growth_rate, 
            init_features=args.init_features,
            rec_features=args.rec_features,
            bn_size=args.bn_size,
            drop_rate=args.drop_rate,
            bottleneck=False).to(args.device)

    model.load_state_dict(model_state_dict)

    # Set up cylinder training and testing loaders
    log.log('Setting up cylinder loaders.')
    testing_dir = '../../step-testing/'
    ntest = np.array([0,8,16]).astype(int)
    u0 = np.linspace(7500,47500,17)/5000.0

    stepLoader = BackwardStepLoader(testing_dir, testing_dir, shuffle=False)
    stepLoader.setNormalizingParams(model)
    testing_loader = stepLoader.createTestingLoader(ntest, u0, inUpscale=(1.34,1.34), batch_size=args.test_batch_size)

    with torch.no_grad():
        log.log('Testing model.')
        tic = time.time()
        yPred, yTarget, yInput = modelPred(args, model, testing_loader, log, samples=3, stride=1, tmax=40)
        print(time.time()-tic)
    
    tmax = 6
    for i in range(ntest.shape[0]):
        log.log('Plotting prediction {:d}'.format(i))
        plotPredictionSeries(yInput, yPred, yTarget, bidx=i, nsteps=tmax, stride=6, nsamp=3)
