from torch.utils.data import Dataset
import torch
import numpy as np
import os


class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, shapedata, normalization=True, dummy_node=True):
        
        self.shapedata = shapedata
        self.normalization = normalization
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(root_dir, 'paths_'+points_dataset+'.npy'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        basename = self.paths[idx]
        
        verts_init = np.load(os.path.join(self.root_dir, 'points'+'_'+self.points_dataset, basename+'.npy'))
        if self.normalization:
            verts_init = verts_init - self.shapedata.mean
            verts_init = verts_init/self.shapedata.std
        verts_init[np.where(np.isnan(verts_init))]=0.0
        
        verts_init = verts_init.astype('float32')
        if self.dummy_node:
            verts = np.zeros((verts_init.shape[0]+1, verts_init.shape[1]), dtype=np.float32)
            verts[:-1, :] = verts_init
            verts_init = verts
        verts = torch.Tensor(verts_init)
        

        sample = {'points': verts}

        return sample
    
  