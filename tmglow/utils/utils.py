'''
Random utilities
===
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
===
'''
from zipfile import ZipFile
from nn.tmGlow import TMGlow
import zipfile
import os, errno, copy
import json, sys
import argparse
import torch
import subprocess

PARAM_BLACKLIST = ['epoch_start', 'epochs', 'run_dir', 'ckpt_dir', 'pred_dir']

def toNumpy(tensor):
    '''
    Converts Pytorch tensor to numpy array
    '''
    return tensor.detach().cpu().numpy()

def toTuple(a):
    '''
    Converts array to tuple
    '''
    try:
        return tuple(toTuple(i) for i in a)
    except TypeError:
        return a

def getGpuMemoryMap():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def saveWorkspace(args, model, optimizer, file_name="nsWorkspace", file_id=0):
    '''
    Saves parameters of the model, optimizer, scheduler
    and program arguements into a zip file
    Args:
        args (argparse): object with programs arguements
        model: Neural network model
        optimizer: PyTorch optimizer
        file_id (int): Current epoch or desired ID number for saved workspace
    '''
    # Create state dict of both the model and optimizer
    print('[SaveWorkspace] Saving PyTorch model to file.')
    model_file_name = os.path.join(args.ckpt_dir,'torchModel{:d}.pth'.format(file_id))
    state = {'epoch': file_id, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
    torch.save(state, model_file_name)

    # Write args JSON file
    print('[SaveWorkspace] Saving arguements to file.')
    args_file_name = os.path.join(args.ckpt_dir,"args.json")
    # Get param dictionary and delete params that shouldnt be saved
    args_dict = copy.deepcopy(vars(args))
    del args_dict['device']
    with open(args_file_name, 'w') as args_file:
        json.dump(args_dict, args_file, indent=4)

    # Create a ZipFile Object
    print('[SaveWorkspace] Compressing workspace to zip file.')
    zip_file_name = os.path.join(args.ckpt_dir, file_name+'{:d}.zip'.format(file_id))
    with ZipFile(zip_file_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipObj:
        # Add multiple files to the zip
        zipObj.write(model_file_name, os.path.basename(model_file_name))
        zipObj.write(args_file_name, os.path.basename(args_file_name))
    # Remove independent model and args file
    os.remove(model_file_name)
    os.remove(args_file_name)
    print('[SaveWorkspace] Saved workspace.')

def loadWorkspace(args, file_dir, file_name="nsWorkspace", file_id=0):
    '''
    Loads parameters of the model, optimizer, scheduler
    and program arguements from zip file
    Args:
        file_dir (string): Directory of workspace zip file 
        file_name (string): File name of workspace (default: nsWorkspace)
        file_id (int): Current epoch or desired ID number for saved workspace
    Returns:
        args (argparse): object with programs arguements
        model_state_dict (dict): PyTorch state dictionary of model parameters
        optimizer_state_dict (dict): PyTorch state dictionary of optimizers parameters
    '''
    try:
        print('[LoadWorkspace] Looking for workspace: {}'.format(file_name+"{:d}.zip".format(file_id)))
        zipObj = ZipFile(os.path.join(file_dir, file_name+"{:d}.zip".format(file_id)))
    except FileNotFoundError:
        print('[LoadWorkspace] Could note find workspace zip file!')
        print(os.path.join(file_dir, file_name+"{:d}.zip".format(file_id)))
        return

    # Extract files to same folder as zip
    print('[LoadWorkspace] Workspace found, extracting contents.')
    zipObj.extractall(file_dir)

    # First load the arguements
    try:
        json_file = os.path.join(file_dir, "args.json")
        with open(json_file) as json_file_data:
            loaded_json = json.loads(json_file_data.read())
            for x in loaded_json:
                if(not x in PARAM_BLACKLIST): # Check param black list for params to not overwrite
                    setattr(args, x, loaded_json[x])
            # Delete json file
            os.remove(json_file)
    except FileNotFoundError:
        print('[LoadWorkspace] No JSON file found for arguements!')
    except:
        print('[LoadWorkspace] Failed arg JSON load/read.')
        print(sys.exc_info())


    # Load state dicts
    try:
        torch_file = os.path.join(file_dir,'torchModel{:d}.pth'.format(file_id))
        param_dict = torch.load(torch_file, map_location=lambda storage, loc: storage)
        os.remove(torch_file)
    except FileNotFoundError:
        print('[LoadWorkspace] Could not find PyTorch model!')

    # Split up the state dict.
    model_state_dict = param_dict['state_dict']
    optimizer_state_dict = param_dict['optimizer']
    print('[LoadWorkspace] Done loading workspace...')

    return args, model_state_dict, optimizer_state_dict


def modelPred(args, model, testing_loader, log, samples=1, stride=1, tmax=1):
    '''
    Tests the model on a single GPUs, used for plotting
    Args:
        model (torch.nn.Module): PyTorch model to test
        stride (int): The stride the low-fidelity input takes compared to output
        samples (int): Number of prediction to sample from the model
    Returns:
        ypred (torch.Tensor): tensor of un-normalized predicted  values
        ytarget (torch.Tensor): tensor of un-normalized target values of test cases
        yinput (torch.Tensor): tensor of un-normalized input values
    '''
    model.eval()
    batch_index = 0
    ypred = None
    ytarget = None

    in_std = model.in_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    in_mu = model.in_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    out_std = model.out_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    out_mu = model.out_mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    for mbIdx, (input0, target0, u0) in enumerate(testing_loader):
        log.log('Running mini-batch {:d}/{:d}'.format(mbIdx+1, len(testing_loader)))

        u0 = u0.to(args.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        u0 = torch.cat((u0,u0,u0**2), dim=2)
        input = input0.to(args.device)
        target0 = u0*(out_std*target0.to(args.device) + out_mu)

        # If the first mini-batch we should set up the tensors
        # to save the models predictions
        if(mbIdx == 0):
            yinput = (u0*(in_std*input[:,:,:3]+in_mu)).cpu()
            ytarget = target0.cpu()

            dims = list(ytarget.size())
            dims[0] = len(testing_loader.dataset) # expand batch dim to full test set
            dims = [samples]+dims  
            dims[2] = (tmax)//stride # Expand for temporal super-position
            ypred = torch.zeros(dims).type(ytarget.type()).cpu()
        else:
            yinput = torch.cat((yinput, (u0*(in_std*input[:,:,:3]+in_mu)).cpu()), dim=0)
            ytarget = torch.cat((ytarget, target0.cpu()), dim=0)

        # Temp tensor to store predictions of mini-batch
        dims = [samples]+list(target0.size())
        dims[2] = (tmax)//stride # Expand for temporal super-position 
        ypred_mb = torch.zeros(dims).type(input.type()) + 10000

        # Loop through samples
        for i in range(samples):
            log.log('Running sample {:d}.'.format(i))
            if( isinstance(model, TMGlow) ):
                hKey = model.initLSTMStates(torch.LongTensor(input.size(0)).random_(0, int(1e8)), [target0.size(-2), target0.size(-1)])
            else:
                hKey = model.module.initLSTMStates(torch.LongTensor(input.size(0)).random_(0, int(1e8)), [target0.size(-2), target0.size(-1)])
            h0 = copy.deepcopy(hKey)
            # h0 = None # Set recurrent params to None in beginning
            # Loop of time-steps
            for tstep in range(0, tmax):

                # Model forward
                input0 = input[:,tstep]
                ypred0, logp, h0 = model.sample(input0, h0)

                if(tstep%stride == 0):
                    yPredHat = u0.squeeze(1)*(out_std*ypred0 + out_mu)
                    ypred_mb[i,:,tstep//stride] = yPredHat.detach()

                if(tstep%20 == 0):
                    for j in range(len(h0)):
                        h_out, c_out = h0[j]
                        h_key, c_key = hKey[j]
                        h0[j] = (0.5*h_out.detach() + 0.5*h_key, 0.5*c_out.detach() + 0.5*c_key)

        # Transfer predicted values off GPU
        ypred[:,batch_index:batch_index+ypred_mb.size(1)] = ypred_mb.cpu()
        batch_index = batch_index + ypred_mb.size(1)
        log.log('Number of elements unset: {}'.format(torch.sum(ypred > 10000)))



    # Return predicted values
    return ypred, ytarget, yinput