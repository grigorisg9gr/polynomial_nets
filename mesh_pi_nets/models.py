import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math


class SpiralPoly(nn.Module):
    def __init__(self, in_c, 
                 spiral_size,
                 out_c, 
                 activation='elu', 
                 bias=True, 
                 device=None, 
                 injection = False, 
                 residual = False, 
                 num_points = None, 
                 order = 1, 
                 normalize = 'final', 
                 model = 'full'):
        super(SpiralPoly, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.device = device
        self.injection = injection
        self.residual = residual
        self.order = order
        self.normalize = normalize
        self.model = model

        if bias:
            # constant parameter of the polynomial
            self.bias = Parameter(torch.Tensor(out_c))
            bound = 1 / math.sqrt(in_c * spiral_size)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        if self.injection:
            if self.model == 'full':
            # conva and convs correspond to A and S matrices of the NCP decomposition (the constant matrices B are abosorbed by A)
                self.conva =  nn.ModuleList([nn.Linear(in_c*spiral_size,out_c,bias=False) for k in range(0,self.order-1)])
                self.convs =  nn.ModuleList([nn.Linear(in_c*spiral_size,out_c,bias=False)] + 
                                            [nn.Linear(out_c*spiral_size,out_c,bias=False) for k in range(0,self.order-2)])
            elif self.model == 'simple':
                # this is the compact version of the model, where we use a single parameter matrix and perform element-wise multiplications
                self.convs = nn.Linear(in_c*spiral_size,out_c,bias=False)  
                
            if self.normalize == 'final':
                # normalise only the higher order term of the entire polynomial
                self.normalizer =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)
            else:
                # normalise once for every recursion step
                self.normalizer =  nn.ModuleList([nn.BatchNorm1d((num_points+1) * out_c, affine= True)
                                                  for k in range(0,self.order-1)])   
        else:
            if self.model == 'mlp':
                self.fc1 = nn.Linear(in_c*spiral_size, out_c, bias = True)
                self.fc2 = nn. Linear(out_c, out_c, bias = True)
            elif self.model == 'linear':
                self.conv = nn.Linear(in_c * spiral_size, out_c, bias=False)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(bsize * num_pts * spiral_size)  # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1, 1).repeat([1, num_pts * spiral_size]).view(
            -1).long()  # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(bsize * num_pts,
                                                        spiral_size * feats)  # [bsize*numpt, spiral*feats]

        if self.injection:
            if self.model == 'full':
                spirals_x1 = spirals  # input z
                spirals_xk_1 = spirals  # output of previous order term xk_1
                for k in range(0, self.order - 1):
                    a_xk = self.conva[k](spirals_x1)
                    s_xk = self.convs[k](spirals_xk_1)

                    if self.normalize == 'all':
                        # normalise the output of the recursion
                        if k >= 1 and self.residual:
                            out_k = a_xk * s_xk + a_xk + out_k
                        else:
                            out_k = a_xk * s_xk + a_xk
                        out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                        out_k = self.normalizer[k](out_k).reshape(bsize * num_pts, -1)
                        spirals_xk_1 = out_k.view(bsize, num_pts, self.out_c) \
                            [batch_index, spirals_index, :].view(bsize * num_pts, spiral_size * self.out_c)
                    elif self.normalize == '2nd':
                        # normalise only the 2nd order term in the recursive formulation
                        if k >= 1 and self.residual:
                            out_k_1 = out_k
                            out_k = a_xk * s_xk
                            out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                            out_k = self.normalizer[k](out_k).reshape(bsize * num_pts, -1) + a_xk + out_k_1
                        else:
                            out_k = a_xk * s_xk
                            out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                            out_k = self.normalizer[k](out_k).reshape(bsize * num_pts, -1) + a_xk
                        spirals_xk_1 = out_k.view(bsize, num_pts, self.out_c) \
                            [batch_index, spirals_index, :].view(bsize * num_pts, spiral_size * self.out_c)
                    elif self.normalize == 'final':
                        # normalise only the higher order term of the entire polynomial
                        if k >= 1 and self.residual:
                            if k == self.order - 2:
                                out_k_1 = out_k
                                out_k = a_xk * s_xk
                                out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                                out_k = self.normalizer(out_k).reshape(bsize * num_pts, -1) + a_xk + out_k_1
                            else:
                                out_k = a_xk * s_xk + a_xk + out_k
                        else:
                            if k == self.order - 2:
                                out_k = a_xk * s_xk
                                out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                                out_k = self.normalizer(out_k).reshape(bsize * num_pts, -1) + a_xk
                            else:
                                out_k = a_xk * s_xk + a_xk
                        spirals_xk_1 = out_k.view(bsize, num_pts, self.out_c) \
                            [batch_index, spirals_index, :].view(bsize * num_pts, spiral_size * self.out_c)

                out_feat = out_k + self.bias


            elif self.model == 'simple':

                out_1 = self.convs(spirals)
                out_feat = out_1
                for k in range(0, self.order - 1):
                    out_k = out_1 ** (k + 2)
                    if self.normalize == '2nd':
                        out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                        out_k = self.normalizer[k](out_k).reshape(bsize * num_pts, -1)
                    elif self.normalize == 'final':
                        if k == self.order - 2:
                            out_k = out_k.reshape(bsize, num_pts * out_k.shape[-1])
                            out_k = self.normalizer(out_k).reshape(bsize * num_pts, -1)
                    out_feat = out_k + out_feat

                out_feat = out_feat + self.bias

        else:
            if self.model == 'mlp':
                out_feat = self.fc2(nn.ReLU()(self.fc1(spirals)))
            elif self.model == 'linear':
                out_feat = self.conv(spirals) + self.bias

        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class SpiralPolyAE(nn.Module):
    def __init__(self,
                 filters_enc,
                 filters_dec,
                 latent_size,
                 mesh_sizes,
                 spiral_sizes,
                 spirals,
                 D, U,
                 device,
                 activation='elu',
                 injection=False,
                 residual=False,
                 order=1,
                 normalize=None,
                 model=None):
        super(SpiralPolyAE, self).__init__()

        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spirals = spirals
        self.mesh_sizes = mesh_sizes
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.conv = []
        
        input_size = filters_enc[0][0]
        for i in range(len(filters_enc)):
            for j in range(1, len(filters_enc[i])):
                self.conv.append(SpiralPoly(input_size, spiral_sizes[i], filters_enc[i][j],
                                            activation=activation, device=device,
                                            injection=injection, residual=residual,
                                            num_points=mesh_sizes[i], order=order,
                                            normalize=normalize, model=model).to(device))
                input_size = filters_enc[i][j]
            if i < len(filters_enc) - 1:
                self.conv.append(SpiralPoly(input_size, spiral_sizes[i], filters_enc[i + 1][0],
                                            activation=activation, device=device,
                                            injection=injection, residual=residual,
                                            num_points=mesh_sizes[i], order=order,
                                            normalize=normalize, model=model).to(device))
                input_size = filters_enc[i + 1][0]

        self.conv = nn.ModuleList(self.conv)

        self.fc_latent_enc = nn.Linear((mesh_sizes[-1] + 1) * input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (mesh_sizes[-1] + 1) * filters_dec[0][0])

        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(filters_dec)):
            for j in range(1, len(filters_dec[i])):
                out_activation = 'identity' if i == len(filters_dec) - 1 and j == len(
                    filters_dec[-1]) - 1 else activation
                self.dconv.append(SpiralPoly(input_size, spiral_sizes[-2 - i], filters_dec[i][j],
                                             activation=out_activation, device=device,
                                             injection=injection, residual=residual,
                                             num_points=mesh_sizes[-2 - i], order=order,
                                             normalize=normalize, model=model).to(device))
                input_size = filters_dec[i][j]

            if i < len(filters_dec) - 1:
                out_activation = 'identity' if i == len(filters_dec) - 2 and len(
                    filters_dec[-1]) == 1 else activation
                self.dconv.append(SpiralPoly(input_size, spiral_sizes[-2 - i], filters_dec[i + 1][0],
                                             activation=out_activation, device=device,
                                             injection=injection, residual=residual,
                                             num_points=mesh_sizes[-2 - i], order=order,
                                             normalize=normalize, model=model).to(device))
                input_size = filters_dec[i + 1][0]

        self.dconv = nn.ModuleList(self.dconv)

    def encode(self, x):
        bsize = x.size(0)
        j = 0
        for i in range(len(self.filters_enc) - 1):
            for k in range(len(self.filters_enc[i])):
                x = self.conv[j](x, self.spirals[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(self.D[i], x)

        for k in range(1, len(self.filters_enc[-1])):
            x = self.conv[j](x, self.spirals[len(self.filters_enc) - 1].repeat(bsize, 1, 1))
            j += 1

        x = x.view(bsize, -1)
        out = self.fc_latent_enc(x)
        return out

    def decode(self, z):
        bsize = z.shape[0]

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.mesh_sizes[-1] + 1, -1)

        j = 0
        for i in range(len(self.filters_dec) - 1):
            x = torch.matmul(self.U[-1 - i], x)
            for k in range(len(self.filters_dec[i])):
                x = self.dconv[j](x, self.spirals[-2 - i].repeat(bsize, 1, 1))
                j += 1

        for k in range(1, len(self.filters_dec[-1])):
            x = self.dconv[j](x, self.spirals[0].repeat(bsize, 1, 1))
            j += 1

        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x
