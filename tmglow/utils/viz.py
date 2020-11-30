'''
Visualization functions used during training
===
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
===
'''
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import os

def plotVelocityPred(args, input0, yPred0, yTarget0, bidx=0, stride=1, epoch=0):
    '''
    Used to plot predictions of the model during training
    Args:
        yPred0 (torch.Tensor): [samp x mb x t x w x h] model prediction
        uTarget (torch.Tenosr): Target
    '''
    input0 = input0.detach().cpu().numpy()
    yPred = yPred0.detach().cpu().numpy()
    yTarget = yTarget0.detach().cpu().numpy()
    nsteps = 10

    cmap0 = 'inferno'
    for i, field in enumerate(['ux','uy','p']):
        plt.close("all")

        fig, ax = plt.subplots(5, 10, figsize=(2.5*yPred0.size(1), 5))
        fig.subplots_adjust(wspace=0.5)

        for t0 in range(nsteps):
            ax[0, int(t0)].imshow(input0[bidx,t0*stride,i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower')

        for t0 in range(nsteps):
            
            c_max = max([np.amax(yTarget[bidx,:,i,:,:])])
            c_min = min([np.amin(yTarget[bidx,:,i,:,:])])
            ax[1, t0].imshow(yTarget[bidx, t0*stride ,i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=c_min)
            # Plot sampled predictions
            ax[2, t0].imshow(yPred[0, bidx, t0*stride, i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=c_min)
            ax[3, t0].imshow(yPred[1, bidx, t0*stride, i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=c_min)
            pcm = ax[4, t0].imshow(np.var(yPred[:, bidx, t0*stride, i,:,:], axis=0), extent=[0,1,0,1], cmap=cmap0, origin='lower')
            fig.colorbar(pcm, ax=ax[4, t0], shrink=0.6)

        for ax0 in ax.flatten().tolist():
            ax0.set_xticks([])
            ax0.set_yticks([])

        file_dir = args.pred_dir
        # If director does not exist create it
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = file_dir+"/{:s}_pred{:d}_epoch{:d}".format(field, bidx, epoch)
        plt.savefig(file_name+".png", bbox_inches='tight')

def plotNumericalPred(args, yPred0, yTarget0, nut0, bidx=0, tstep=0, epoch=0 ):
    '''
    Used to plot predictions of the model during training
    Args:
        yPred0 (torch.Tensor): [samp x mb x t x w x h] model prediction
        uTarget (torch.Tenosr): Target
    '''
    yPred = yPred0.detach().cpu().numpy()
    yTarget = yTarget0.detach().cpu().numpy()
    nut = nut0.detach().cpu().numpy()
    nsteps = 10

    cmap0 = 'inferno'
    for i, field in enumerate(['ux','uy','p']):
        plt.close("all")

        # fig, ax = plt.subplots(2+yPred0.size(0), yPred0.size(2), figsize=(2*yPred0.size(1), 3+3*yPred0.size(0)))
        fig, ax = plt.subplots(1, 4, figsize=(8, 4))
        fig.subplots_adjust(wspace=0.5)


        c_max = max([np.amax(yTarget[bidx,i,:,:]), np.amax(yPred[bidx,i,1:-1,1:-1])])
        c_min = min([np.amin(yTarget[bidx,i,:,:]), np.amin(yPred[bidx,i,1:-1,1:-1])])
        c_max = max([np.amax(yTarget[bidx,i,:,:])])
        c_min = min([np.amin(yTarget[bidx,i,:,:])])

        ax[0].imshow(yTarget[bidx, i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=c_min)
        # Plot sampled predictions
        ax[1].imshow(yPred[bidx, i,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower', vmax=c_max, vmin=c_min)
        pcm = ax[2].imshow(np.power(yTarget[bidx,i,1:-1,1:-1] - yPred[bidx,i,1:-1,1:-1], 2), extent=[0,1,0,1], cmap=cmap0, origin='lower')
        fig.colorbar(pcm, ax=ax[2], shrink=0.6)
        pcm = ax[3].imshow(nut[bidx,0,:,:], extent=[0,1,0,1], cmap=cmap0, origin='lower')
        fig.colorbar(pcm, ax=ax[3], shrink=0.6)

        for ax0 in ax.flatten().tolist():
            ax0.set_xticks([])
            ax0.set_yticks([])

        file_dir = args.pred_dir
        # If director does not exist create it
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = file_dir+"/numericalPred{:s}_pred{:d}_epoch{:d}_tstep{:d}".format(field, bidx, epoch, tstep)
        plt.savefig(file_name+".png", bbox_inches='tight')
