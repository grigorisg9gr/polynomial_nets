import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from datetime import datetime
import json

def save_config(config, shared_path):
    save_path = os.path.join(shared_path, "params.json")

    with open(save_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_time():
    return datetime.now().strftime("%d%m_%H%M%S")


def main(config):
    cudnn.benchmark = True

    config.time_now = get_time()

    if config.mode == 'train':
        data_loader = get_loader(data_path=config.data_path,
                                 batch_size=config.batch_size,
                                 mode=config.mode,
                                 num_workers=config.num_workers)


        shared_path = '{}{}'.format(config.base_path, config.time_now)

        config.base_path = shared_path
        config.model_path = "{}/{}".format(shared_path, config.model_path)
        config.sample_path = "{}/{}".format(shared_path, config.sample_path)
        config.logs_path = "{}/{}".format(shared_path, config.logs_path)
        config.validation_path = "{}/{}".format(shared_path, config.validation_path)
        
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        if not os.path.exists(config.sample_path):
            os.makedirs(config.sample_path)

        if not os.path.exists(config.logs_path):
            os.makedirs(config.logs_path)

        if not os.path.exists(config.validation_path):
            os.makedirs(config.validation_path)

        save_config(config, shared_path)

    else:
        data_loader = None


    # for i in range(400, 705, 100):

        # config.seed = i
    solver = Solver(config, data_loader)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test' or config.mode == 'sample':
        data_loader = None
        if config.mode == 'test':
            solver.test()
        else:
            solver.sample(config.n_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_size', type=int, default=100) 
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--g_layers', type=int, nargs='+', default=[100, 512, 256, 128, 64, 3])  # network structure of the Generator. First layer (100) is the size of the noise, the last layer (3) is the output size (RGB image in our case)
    parser.add_argument('--d_layers', type=int, nargs='+', default=[3, 64, 128, 256, 512, 1])  # network structure of the Discriminator. First layer (3) is the input size (RGB image in our case) and the last layer (1) corresponds to the output
    parser.add_argument('--activation_fn', type=str2bool, default=True)
    parser.add_argument('--inject_z', type=str2bool, default=True)
    parser.add_argument('--concat_injection', type=str2bool, default=False)  # whether the injection will be concatenated or multiplied
    parser.add_argument('--loss', type=str, default='original')  # can be changed to Wassterstein GAN with GP. Just put 'wgan-gp'
    parser.add_argument('--gp_weight', type=float, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pc_name', type=str, default='neumann')
    parser.add_argument('--spectral_norm', type=str2bool, default=False)
    parser.add_argument('--transform_z', type=str2bool, default=False)  # whether to have an MLP before injection
    parser.add_argument('--transform_rep', type=int, default=1)  # if transform_z is ``True'', how many MLPs to have in a row before injection
    parser.add_argument('--norm', type=str, default='batch')  # can be changed to ``instance'' to activate the instance normalization

    # misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--logs_path', type=str, default="logs")
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--base_path', type=str, default='./dcgan_inject/')  # where to save the experiment results
    parser.add_argument('--ckpt_gen_path', type=str, default='')
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=350)
    parser.add_argument('--validation_step', type=int, default=350)
    parser.add_argument('--validation_path', type=str, default="validation")
    parser.add_argument('--cifar10_path', type=str, default='./cifar_10_stats.pkl')  # the path for the cifar_10_stats.pkl file
    parser.add_argument('--FID_images', type=int, default=2048)
    parser.add_argument('--num_imgs_val', type=int, default=2048)
    parser.add_argument('--max_score', type=float, default=0)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=100000)  # number of samples to generate under the ``sample'' scenario
    config = parser.parse_args()
    print(config)
    main(config)
