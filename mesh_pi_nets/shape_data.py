### Code obtained and modified from https://github.com/anuragranj/coma, Copyright (c) 2018 Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, Michael J. Black and the Max Planck Gesellschaft

import os
import numpy as np

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh

import trimesh


class ShapeData(object):
    def __init__(self, nVal,
                 train_file,
                 test_file,
                 reference_mesh_file,
                 normalization=True,
                 meshpackage='mpi-mesh',
                 load_flag=True,
                 mean_subtraction_only=False):
        self.nVal = nVal
        self.train_file = train_file
        self.test_file = test_file
        self.vertices_train = None
        self.vertices_val = None
        self.vertices_test = None
        self.n_vertex = None
        self.n_features = None
        self.normalization = normalization
        self.meshpackage = meshpackage
        self.load_flag = load_flag
        self.mean_subtraction_only = mean_subtraction_only

        if self.load_flag:
            self.load()
        if self.meshpackage == 'trimesh':
            self.reference_mesh = trimesh.load(reference_mesh_file, process=False)
        elif self.meshpackage == 'mpi-mesh':
            self.reference_mesh = Mesh(filename=reference_mesh_file)

        if self.load_flag:
            self.mean = np.mean(self.vertices_train, axis=0)
            self.std = np.std(self.vertices_train, axis=0)
        else:
            self.mean = None
            self.std = None
        self.normalize()

    def load(self):
        vertices_train = np.load(self.train_file)
        self.vertices_train = vertices_train[:-self.nVal]
        self.vertices_val = vertices_train[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]
        self.n_features = self.vertices_train.shape[2]

        if os.path.exists(self.test_file):
            self.vertices_test = np.load(self.test_file)
            self.vertices_test = self.vertices_test

    def normalize(self):
        if self.load_flag:
            if self.normalization:
                if self.mean_subtraction_only:
                    self.std = np.ones_like((self.std))
                self.vertices_train = self.vertices_train - self.mean
                self.vertices_train = self.vertices_train / self.std
                self.vertices_train[np.where(np.isnan(self.vertices_train))] = 0.0

                self.vertices_val = self.vertices_val - self.mean
                self.vertices_val = self.vertices_val / self.std
                self.vertices_val[np.where(np.isnan(self.vertices_val))] = 0.0

                if self.vertices_test is not None:
                    self.vertices_test = self.vertices_test - self.mean
                    self.vertices_test = self.vertices_test / self.std
                    self.vertices_test[np.where(np.isnan(self.vertices_test))] = 0.0

                self.N = self.vertices_train.shape[0]

                print('Vertices normalized')
            else:
                print('Vertices not normalized')

    def save_meshes(self, filename, meshes, mesh_indices):
        for i in range(meshes.shape[0]):
            if self.normalization:
                vertices = meshes[i].reshape((self.n_vertex, self.n_features)) * self.std + self.mean
            else:
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))
            if self.meshpackage == 'trimesh':
                new_mesh = self.reference_mesh
                if self.n_features == 3:
                    new_mesh.vertices = vertices
                elif self.n_features == 6:
                    new_mesh.vertices = vertices[:, 0:3]
                    colors = vertices[:, 3:]
                    colors[np.where(colors < 0)] = 0
                    colors[np.where(colors > 1)] = 1
                    vertices[:, 3:] = colors
                    new_mesh.visual = trimesh.visual.create_visual(vertex_colors=vertices[:, 3:])
                else:
                    raise NotImplementedError
                new_mesh.export(filename + '.' + str(mesh_indices[i]).zfill(6) + '.ply', 'ply')
            elif self.meshpackage == 'mpi-mesh':
                if self.n_features == 3:
                    mesh = Mesh(v=vertices, f=self.reference_mesh.f)
                    mesh.write_ply(filename + '.' + str(mesh_indices[i]).zfill(6) + '.ply')
                else:
                    raise NotImplementedError
        return 0