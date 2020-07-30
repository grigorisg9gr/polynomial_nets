import torch
import torch.nn as nn
import os
from torch import optim
from torch.autograd import Variable
from model import Discriminator
from model import Generator
from logger import Logger
import numpy as np
from torchvision.utils import save_image
from torch.autograd import grad as torch_grad
from collections import OrderedDict
from IS.inception_score import inception_score as IS 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import json
from FID.fid_score import calculate_fid_given_paths as FID

def save_is(base_path, score):
    save_file = os.path.join(base_path, "scores.json")

    with open(save_file, 'a+') as fp:
        json.dump(score, fp, indent=4)
        fp.write("\n")
        fp.close()


def denorm(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_numpy(x):
    if torch.cuda.is_available():
        x = x.data.cpu()
    return x.numpy()


class Solver(object):
    def __init__(self, config, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.pc_name = config.pc_name
        self.base_path = config.base_path
        self.time_now = config.time_now
        self.inject_z = config.inject_z
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.sample_size = config.sample_size
        self.logs_path = config.logs_path
        self.save_every = config.save_every
        self.activation_fn = config.activation_fn
        self.max_score = config.max_score
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validation_step = config.validation_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.g_layers = config.g_layers
        self.d_layers = config.d_layers
        self.z_dim = self.g_layers[0]
        self.num_imgs_val = config.num_imgs_val
        self.criterion = nn.BCEWithLogitsLoss()
        self.ckpt_gen_path = config.ckpt_gen_path
        self.gp_weight = config.gp_weight
        self.loss = config.loss
        self.seed = config.seed
        self.validation_path = config.validation_path
        self.FID_images = config.FID_images
        self.transform_rep = config.transform_rep
        self.transform_z = config.transform_z
        self.spectral_norm = config.spectral_norm
        self.cifar10_path = config.cifar10_path
        self.fid_score = 100000
        self.concat_injection = config.concat_injection
        self.norm = config.norm
        self.build_model()

    def build_model(self):
        torch.manual_seed(self.seed)
        self.generator = Generator(g_layers=self.g_layers, activation_fn=self.activation_fn, inject_z=self.inject_z, transform_rep=self.transform_rep, transform_z=self.transform_z, concat_injection=self.concat_injection, norm=self.norm)
        self.discriminator = Discriminator(d_layers=self.d_layers, activation_fn=self.activation_fn, spectral_norm=self.spectral_norm)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        self.g_optimizer = optim.Adam(self.generator.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.logger = Logger(self.logs_path)


        self.gen_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.disc_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        self.total_params = self.gen_params + self.disc_params

        print("Generator params: {}".format(self.gen_params))
        print("Discrimintor params: {}".format(self.disc_params))
        print("Total params: {}".format(self.total_params))

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def reset_grad(self):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, height, width),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight*((gradients_norm - 1) ** 2).mean()

    def train(self):
        total_step = len(self.data_loader)
        for epoch in range(self.num_epochs):
            for i, (data, _) in enumerate(self.data_loader):

                batch_size = data.size(0)
                # train Discriminator
                data = data.type(torch.FloatTensor)
                data = to_cuda(data)

                real_labels = to_cuda(torch.ones(batch_size, self.d_layers[-1]))
                fake_labels = to_cuda(torch.zeros(batch_size, self.d_layers[-1]))

                outputs_real = self.discriminator(data)
                z = to_cuda(torch.randn(batch_size, self.z_dim, 1, 1))
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data)

                if self.loss == 'original':
                    d_loss_real = self.criterion(outputs_real.squeeze(), real_labels.squeeze())
                    d_loss_fake = self.criterion(outputs_fake.squeeze(), fake_labels.squeeze())
                    d_loss = d_loss_real + d_loss_fake

                elif self.loss == 'wgan-gp':
                    gradient_penalty = self.gradient_penalty(data, fake_data)
                    d_loss = - outputs_real.mean() + outputs_fake.mean() + gradient_penalty

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # train Generator
                z = to_cuda(torch.randn(batch_size, self.z_dim, 1, 1))
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data)

                if self.loss == 'original':
                    g_loss = self.criterion(outputs_fake.squeeze(), real_labels.squeeze())
                elif self.loss == 'wgan-gp':
                    g_loss = -outputs_fake.mean()

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if (i+1) % self.log_step == 0:
                    print('Epoch [{0:d}/{1:d}], Step [{2:d}/{3:d}], d_real_loss: {4:.4f}, '
                          ' g_loss: {5:.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, d_loss.item(),
                                                   g_loss.item()))

                    # log scalars in tensorboard
                    info = {
                        'd_real_loss': d_loss.item(),
                        'g_loss': g_loss.item(),
                        'inception_score': self.max_score
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, epoch*total_step + i + 1)

                if (i + 1) % self.sample_step == 0:
                    save_image(denorm(fake_data).cpu(), self.sample_path + "/epoch_{}_{}.png".format(i + 1, epoch + 1), nrow=8)

                if (i + 1) % self.validation_step == 0:
                    fake_data_all = np.zeros((self.num_imgs_val, fake_data.size(1), fake_data.size(2), fake_data.size(3)))
                    for j in range(self.num_imgs_val // batch_size):
                        fake_data_all[j*batch_size:(j+1)*batch_size] = to_numpy(fake_data)
                    npy_path = os.path.join(self.model_path, '{}_{}_val_data.pkl'.format(epoch + 1, i + 1))
                    np.save(npy_path, fake_data_all)
                    score, _ = IS(fake_data_all, cuda=True, batch_size=batch_size)
                    if score > self.max_score:
                        print("Found new best IS score: {}".format(score))
                        self.max_score = score
                        data = "IS " + str(self.seed) + " " + str(epoch + 1) + " " + str(i + 1) + " " + str(self.max_score)
                        save_is(self.base_path, data)
                        g_path = os.path.join(self.model_path, 'generator-best.pkl')
                        d_path = os.path.join(self.model_path, 'discriminator-best.pkl')
                        torch.save(self.generator.state_dict(), g_path)
                        torch.save(self.discriminator.state_dict(), d_path)
                    for j in range(self.FID_images):
                        z = to_cuda(torch.randn(1, self.z_dim, 1, 1))
                        fake_datum = self.generator(z)
                        save_image(denorm(fake_datum.squeeze()).cpu(), self.validation_path + "/" + str(j) + ".png")
                    fid_value = FID([self.validation_path, self.cifar10_path], 64, True, 2048)
                    if fid_value < self.fid_score:
                        self.fid_score = fid_value
                        print("Found new best FID score: {}".format(self.fid_score))
                        data = "FID " + str(self.seed) + " " + str(epoch + 1) + " " + str(i + 1) + " " + str(self.fid_score)
                        save_is(self.base_path, data)
                        g_path = os.path.join(self.model_path, 'generator-best-fid.pkl')
                        d_path = os.path.join(self.model_path, 'discriminator-best-fid.pkl')
                        torch.save(self.generator.state_dict(), g_path)
                        torch.save(self.discriminator.state_dict(), d_path)                        

            if (epoch + 1) % self.save_every == 0:
                g_path = os.path.join(self.model_path, 'generator-{}.pkl'.format(epoch+1))
                d_path = os.path.join(self.model_path, 'discriminator-{}.pkl'.format(epoch+1))
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)


    def sample(self, n_samples):
        self.n_samples = n_samples
        self.generator = Generator(g_layers=self.g_layers, inject_z=self.inject_z)
        self.generator.load_state_dict(torch.load(self.ckpt_gen_path))
        if torch.cuda.is_available():
            self.generator.cuda()
        self.generator.eval()

        z_samples = to_cuda(torch.randn(n_samples, self.z_dim, 1, 1))
        generated_samples = self.generator(z_samples)
        generated_samples = to_numpy(generated_samples)
        np.save('./saved/generated_samples.npy', generated_samples)
        z_samples = to_numpy(z_samples)
        np.save('./saved/z_samples.npy', z_samples)
