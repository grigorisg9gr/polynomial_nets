import argparse

import sys

sys.path.append('../')
import numpy as np
import json
import os
import copy
from shape_data import ShapeData
from mesh_sampling import compute_downsampling

try:
    import psbody.mesh

    found = True
except ImportError:
    found = False
if found:
    pass

from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader

from spiral_utils import get_adj_trigs, generate_spirals
from models import SpiralPolyAE

from test_funcs import test_autoencoder_dataloader
from train_funcs import train_autoencoder_dataloader

import torch
from tensorboardX import SummaryWriter

from sklearn.metrics.pairwise import euclidean_distances

meshpackage = 'trimesh'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list2int(v):
    return [int(c) for c in v.split(',')]


def str2ListOfLists2int(v):
    return [[[] if c == ' ' else int(c) for c in vi.split(',')] for vi in v.split(',,')]


def str2list2float(v):
    return [float(c) for c in v.split(',')]


def str2list2bool(v):
    return [str2bool(c) for c in v.split(',')]


def str2ListOfLists2bool(v):
    return [[[] if c == ' ' else str2bool(c) for c in vi.split(',')] for vi in v.split(',,')]


def loss_l1(outputs, targets):
    L = torch.abs(outputs - targets).mean()
    return L


def main(args):
    ## Set seeds and invoke device

    torch.cuda.get_device_name(args['device_idx'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.benchmark = False
    np.random.seed(args['seed'])
    torch.set_num_threads(args['num_threads'])

    if args['GPU']:
        device = torch.device("cuda:" + str(args['device_idx']) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    ## Set paths

    path = os.path.join(args['root_dir'], args['dataset'])
    args['data'] = os.path.join(path, 'preprocessed', args['name'])
    args['reference_mesh_file'] = os.path.join(path, 'templates/template.obj')
    args['downsample_directory'] = os.path.join(path, 'templates',
                                                args['downsample_method'])
    args['results_folder'] = os.path.join(path, 'results', 'polynomial_autoencoder',
                                          args['downsample_method'],
                                          args['results_folder'],
                                          'latent_' + str(args['nz']))

    ## Create folders
    
    summary_path = os.path.join(args['results_folder'], 'summaries', args['name'])
    checkpoint_path = os.path.join(args['results_folder'], 'checkpoints', args['name'])
    samples_path = os.path.join(args['results_folder'], 'samples', args['name'])
    prediction_path = os.path.join(args['results_folder'], 'predictions', args['name'])
    if not os.path.exists(args['downsample_directory']):
        os.makedirs(args['downsample_directory'])
    
    
    if args['mode'] == 'train':
        if not os.path.exists(os.path.join(args['results_folder'])):
            os.makedirs(os.path.join(args['results_folder']))
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
            
    ## Set reference points (refer to Neural3DMM paper for the discussion on reference points - tl;dr you can set them arbitrarily). Here chosen on the top of the head.

    if args['dataset'] == 'COMA':
        reference_points = [[3567, 4051, 4597]]
    elif args['dataset'] == 'mein3d':
        reference_points = [[23822]]
    elif args['dataset'] == 'DFAUST':
        reference_points = [[414]]

    ## Initialise dataset

    print("Loading data .. ")
    load_flag = True if not os.path.exists(args['data'] + '/mean.npy') or not os.path.exists(
        args['data'] + '/std.npy') else False
    shapedata = ShapeData(nVal=args['nVal'],
                          train_file=args['data'] + '/train.npy',
                          test_file=args['data'] + '/test.npy',
                          reference_mesh_file=args['reference_mesh_file'],
                          normalization=args['data_normalization'],
                          meshpackage=meshpackage, load_flag=load_flag)
    if load_flag:
        np.save(args['data'] + '/mean.npy', shapedata.mean)
        np.save(args['data'] + '/std.npy', shapedata.std)
    else:
        shapedata.mean = np.load(args['data'] + '/mean.npy')
        shapedata.std = np.load(args['data'] + '/std.npy')
        shapedata.n_vertex = shapedata.mean.shape[0]
        shapedata.n_features = shapedata.mean.shape[1]

    ## Load downsampling/upsampling matrices or compute them using the Mesh package (please refer to the Neural3DMM repository for more information)

    M, A, D, U, F = compute_downsampling(args['downsample_directory'],
                                         downsample_method=args['downsample_method'],
                                         shapedata=shapedata,
                                         ds_factors=args['ds_factors'])

    ## Add dummy node to downsampling/upsampling matrices and move the to GPU
    tD = []
    tU = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        d = torch.from_numpy(d).float().to(device)
        u = torch.from_numpy(u).float().to(device)
        tD.append(d)
        tU.append(u)

    ## Compute reference points for downsampled meshes
    print("Calculating reference points for downsampled versions..")
    for i in range(len(args['ds_factors'])):
        if shapedata.meshpackage == 'mpi-mesh':
            dist = euclidean_distances(M[i + 1].v, M[0].v[reference_points[0]])
        elif shapedata.meshpackage == 'trimesh':
            dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    ## Compute local node orderings
    mesh_sizes = [x.v.shape[0] for x in M] if shapedata.meshpackage == 'mpi-mesh' else [x.vertices.shape[0] for x in M]
    Adj, Trigs = get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage=shapedata.meshpackage)
    spirals_np, spiral_sizes, spirals = generate_spirals(args['step_sizes'], M, Adj, Trigs,
                                                         reference_points=reference_points,
                                                         dilation=args['dilation'],
                                                         random=False,
                                                         meshpackage=shapedata.meshpackage,
                                                         counter_clockwise=True)

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    ## Initialise dataloaders

    if args['mode'] == 'train':
        dataset_train = autoencoder_dataset(root_dir=args['data'],
                                            points_dataset='train',
                                            shapedata=shapedata,
                                            normalization=args['data_normalization'])
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args['batch_size'],
                                      shuffle=args['shuffle'],
                                      num_workers=args['num_workers'])
        dataset_val = autoencoder_dataset(root_dir=args['data'],
                                          points_dataset='val',
                                          shapedata=shapedata,
                                          normalization=args['data_normalization'])
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=args['batch_size'],
                                    shuffle=False,
                                    num_workers=args['num_workers'])

    dataset_test = autoencoder_dataset(root_dir=args['data'],
                                       points_dataset=args['test_set'],
                                       shapedata=shapedata,
                                       normalization=args['data_normalization'])
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=args['batch_size'],
                                 shuffle=False,
                                 num_workers=args['num_workers'])

    ## Initialise the model

    model = SpiralPolyAE(filters_enc=args['filter_sizes_enc'],
                         filters_dec=args['filter_sizes_dec'],
                         latent_size=args['nz'],
                         mesh_sizes=mesh_sizes,
                         spiral_sizes=spiral_sizes,
                         spirals=tspirals,
                         D=tD, U=tU,
                         device=device,
                         injection=args['injection'],
                         residual=args['residual'],
                         order=args['order'],
                         normalize=args['normalize'],
                         model=args['model'],
                         activation=args['activation']).to(device)

    ## Initialise optimiser, scheduler and set loss function
    if args['mode'] == 'train':
        optim = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['regularization'])
        if args['scheduler']:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'], gamma=args['decay_rate'])
        else:
            scheduler = None

        if args['loss'] == 'l1':
            loss_fn = loss_l1

    ## parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)

    ############## ----------------  TRAINING LOOP ---------------- ##############
    if args['mode'] == 'train':
        ## configure logging and save hypeparams
        writer = SummaryWriter(summary_path)
        with open(os.path.join(args['results_folder'], 'checkpoints', args['name'] + '_params.json'), 'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp)

        if args['resume']:
            print('loading checkpoint from file %s' % (os.path.join(checkpoint_path, args['checkpoint_file'])))
            checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                                         map_location=device)
            start_epoch = checkpoint_dict['epoch'] + 1
            model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
            optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            print('Resuming from epoch %s' % (str(start_epoch)))
        else:
            start_epoch = 0

        train_autoencoder_dataloader(dataloader_train,
                                     dataloader_val,
                                     device,
                                     model,
                                     optim,
                                     loss_fn,
                                     bsize=args['batch_size'],
                                     start_epoch=start_epoch,
                                     n_epochs=args['num_epochs'],
                                     eval_freq=args['eval_frequency'],
                                     scheduler=scheduler,
                                     writer=writer,
                                     save_recons=args['save_recons'],
                                     shapedata=shapedata,
                                     metadata_dir=checkpoint_path,
                                     samples_dir=samples_path,
                                     checkpoint_path=args['checkpoint_file'])
    elif args['mode'] == 'test':
        print('loading checkpoint from file %s' % (os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar')))
        checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                                     map_location=device)
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        predictions, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device,
                                                                         model,
                                                                         dataloader_test,
                                                                         shapedata,
                                                                         mm_constant=args['mm_constant'])
        np.save(os.path.join(prediction_path, 'predictions'), predictions)

        print('autoencoder: normalized loss', norm_l1_loss)
        print('autoencoder: euclidean distance in mm=', l2_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_threads', type=int, default=1)

    # paths, data etc.
    parser.add_argument('--root_dir', type=str, default='./datasets')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='DFAUST')
    parser.add_argument('--downsample_method', type=str, default='COMA_downsample')
    parser.add_argument('--results_folder', type=str, default='temp')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint')
    parser.add_argument('--data_normalization', type=str2bool, default=True) \
    # multiply with this constant to get your mesh measurements in milimiters (dataset dependent)
    parser.add_argument('--mm_constant', type=int, default=1000)

    # optimisation and training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--eval_frequency', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--regularization', type=float, default=5e-5)
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--decay_steps', type=int, default=1)
    parser.add_argument('--loss', type=str, default='l1')

    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--nVal', type=int, default=100)

    # model hyperparameters
    parser.add_argument('--filter_sizes_enc', type=str2ListOfLists2int, default=[[3], [16], [16], [16], [32]])
    parser.add_argument('--filter_sizes_dec', type=str2ListOfLists2int, default=[[32], [32], [16], [16], [3]])
    parser.add_argument('--nz', type=int, default=16)
    parser.add_argument('--ds_factors', type=str2list2int, default=[4, 4, 4, 4])
    parser.add_argument('--step_sizes', type=str2list2int, default=[1, 1, 1, 1, 1])
    parser.add_argument('--dilation', type=str2list2int, default=None)
    parser.add_argument('--activation', type=str, default='elu')

    # hyperparameters related to the polynomial
    parser.add_argument('--injection', type=str2bool, default=True)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--model', type=str, default='full')
    parser.add_argument('--normalize', type=str, default='final')
    parser.add_argument('--residual', type=str2bool, default=False)

    # misc
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--save_recons', type=str2bool, default=True)  # save reconstructions every epoch
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')  # specify the set on which to test

    # hardware
    parser.add_argument('--GPU', type=str2bool, default=True)
    parser.add_argument('--device_idx', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(vars(args))
