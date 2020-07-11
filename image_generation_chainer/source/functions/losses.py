import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable

def _tensor_to_matrix(tensor, axis=0):
    """ Reshapes a 4-dimensional tensor to 2D matrix. """
    if tensor.ndim == 4:
        # # reshape to a 2D matrix.
        matr = F.reshape(tensor, (tensor.shape[axis], -1))
    elif tensor.ndim == 2:
        matr = tensor
    else:
        matr = F.reshape(tensor, (-1, 1))
    return matr


def cosine_loss(tens1, tens2, absol=True):
    """
    Computes the cosine loss between two representations.
    The cos is computed per element, i.e. assumed that 
    tens1[i] and tens2[i] correspond to the representations
    of which we want to compute the cos.
    Works only on chainer 5.x, because of the einsum.
    """
    mat1 = _tensor_to_matrix(tens1, axis=0)
    mat2 = _tensor_to_matrix(tens2, axis=0)
    # # compute the inner product.
    prod = F.einsum('ij,ij->i', mat1, mat2)
    # # compute the norms.
    norm1 = F.batch_l2_norm_squared(mat1)
    norm2 = F.batch_l2_norm_squared(mat2)
    # # compute the final cosine (per element).
    cos = prod / F.matmul(norm1, norm2)
    if absol:
        # # We restrict the angles to [-90, 90] effectively.
        # # That is, we allow only positive cos.
        cos = F.absolute(cos)
    return F.mean(cos)


def decov_loss(tensor, xp=None, axis=1):
    """
    Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'.
    This version implements the loss in the variable format.

    ARGS:
        axis: (int, optional) If the tensor is 4-dim, it is
            reshaped into 2-dim; axis is the first dimension.
    """
    if xp is None:
        # # get xp module if not provided.
        xp = chainer.cuda.get_array_module(tensor.data)
    # # reshape to a 2D matrix.
    matr = _tensor_to_matrix(tensor, axis=axis)
    # # subtract the mean.
    centered = F.bias(matr, -F.mean(matr))
    # # compute the covariance.
    cov = F.matmul(centered, F.transpose(centered))
    # # compute the frombenius norm.
    frob_norm = F.sum(F.square(cov))
    # # get the norm of diagonal elements.
    # # in chainer 5.x this should work.
#     corr_diag_sqr = F.sum(F.square(F.diagonal(cov1)))
    corr_diag_sqr = F.sum(F.square(cov * xp.eye(cov.shape[0], dtype=cov.dtype)))
    loss = 0.5 * (frob_norm - corr_diag_sqr)
    return loss


def decov_loss_matrix(tensor, xp=None, axis=1):
    """
    Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'.
    This version implements the matrix case, i.e. Variables are converted
    into matrices/tensors.

    ARGS:
        axis: (int, optional) If the tensor is 4-dim, it is
            reshaped into 2-dim; axis is the first dimension.
    """
    if type(tensor, Variable):
        tensor = tensor.data
    if xp is None:
        # # get xp module if not provided.
        xp = chainer.cuda.get_array_module(tensor)
    if tensor.ndim == 4:
        # # reshape to a 2D matrix.
        matr = tensor.reshape((tensor.shape[axis], -1))
    elif tensor.ndim == 2:
        matr = tensor.copy()
    # # subtract the mean.
    mean1 = matr.mean()
    matr -= mean1
    # # compute the covariance.
    cov = xp.matmul(matr, matr.T)
    # # compute the frombenius norm.
    frob_norm = xp.sum(xp.square(cov))
    # # get the norm of diagonal elements.
    corr_diag_sqr = xp.sum(xp.square(xp.diag(cov)))
    loss = 0.5 * (frob_norm - corr_diag_sqr)
    return loss


def loss_revKL_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_revKL_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


# WGAN Loss
def loss_wgan_dis(dis_fake, dis_real):
    loss = - F.mean(dis_real) + F.mean(dis_fake)
    return loss


def loss_wgan_gen(dis_fake):
    loss = - F.mean(dis_fake)
    return loss


def gradient_penalty(dis_output, x):
    grad_x, = chainer.grad([dis_output], [x], enable_double_backprop=True)
    norm_grad_x = F.mean(F.sum(grad_x * grad_x, axis=(1, 2, 3)))
    return norm_grad_x


def gradient_penalty_wgangp(dis_output, x, lipnorm):
    grad_x, = chainer.grad([dis_output], [x], enable_double_backprop=True)
    norm_grad_x = F.sqrt(F.sum(grad_x * grad_x, axis=(1, 2, 3)))
    xp = norm_grad_x.xp
    return F.mean_squared_error(norm_grad_x, lipnorm * xp.ones_like(norm_grad_x.array))
