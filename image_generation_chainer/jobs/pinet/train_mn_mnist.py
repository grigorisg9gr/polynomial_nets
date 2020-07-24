import os, sys
import numpy as np
from socket import gethostname
from time import strftime
import argparse
import chainer
from chainer import training
from chainer.training import extensions, extension
import chainermn
import multiprocessing

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
sys.path.append(base)


from evaluations.extensions import divergence_trainer, sample_generate, sample_generate_conditional
import yaml
import source.yaml_utils as yaml_utils
from source.miscs.model_moiving_average import ModelMovingAverage
from source.misc_train_utils import create_result_dir, load_models, ensure_config_paths, plot_losses_log


def make_optimizer(model, comm, alpha=0.001, beta1=0.9, beta2=0.999, chmn=False, weight_decay_rate=0):
    # # 12/2018: problem in minoas, probably related with openmpi.
    if chmn:
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, 
                                    weight_decay_rate=weight_decay_rate), comm)
    else:
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2,
                                            weight_decay_rate=weight_decay_rate)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='jobs/pinet/fashionmnist_cnn_prodpoly_linear.yml')
    parser.add_argument('--n_devices', type=int)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--results_dir', type=str, default='results_polynomial')
    parser.add_argument('--inception_model_path', type=str,
                        default='/home/user/inception/inception.model')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gen_snapshot', type=str, default=None, help='path to the generator snapshot')
    parser.add_argument('--dis_snapshot', type=str, default=None, help='path to the discriminator snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--multiprocessing', action='store_true', default=False)
    parser.add_argument('--label', type=str, default='synth')
    parser.add_argument('--batch_val', type=int, default=1000)
    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # # ensure that the paths of the config are correct.
    config = ensure_config_paths(config)
    try:
        comm = chainermn.create_communicator(args.communicator)
    except:
        comm = chainermn.create_communicator()
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()
    # # get the pc name, e.g. for chainerui.
    pcname = gethostname()
    print('Init on pc: {}.'.format(pcname))
    if comm.rank == 0:
        print('==========================================')
        print('Using {} communicator'.format(args.communicator))
        print('==========================================')
    gen, dis = load_models(config)
    gen.to_gpu()
    dis.to_gpu()
    mma = ModelMovingAverage(0.999, gen)
    models = {"gen": gen, "dis": dis}
    if args.gen_snapshot is not None:
        print('Loading generator: {}.'.format(args.gen_snapshot))
        chainer.serializers.load_npz(args.gen_snapshot, gen)
    if args.dis_snapshot is not None:
        print('Loading discriminator: {}.'.format(args.dis_snapshot))
        chainer.serializers.load_npz(args.dis_snapshot, dis)
    # Optimizer
    # # convenience function for optimizer:
    func_opt = lambda net, alpha, wdr0=0: make_optimizer(net, comm, chmn=args.multiprocessing,
                                          alpha=alpha, beta1=config.adam['beta1'], 
                                          beta2=config.adam['beta2'], weight_decay_rate=wdr0)
    # Optimizer
    wdr = 0 if 'weight_decay_rate' not in config.updater['args'] else config.updater['args']['weight_decay_rate_gener']
    opt_gen = func_opt(gen, config.adam['alpha'], wdr0=wdr)
    keydopt = 'alphad' if 'alphad' in config.adam.keys() else 'alpha'
    opt_dis = func_opt(dis, config.adam[keydopt])
    opts = {"opt_gen": opt_gen, "opt_dis": opt_dis}
    if hasattr(dis, 'fix_last') and hasattr(dis, 'lin') and dis.fix_last:
        # # This should be used with care. It fixes the linear layer that
        # # makes the classification in the discriminator.
        print('Fixing the linear layer of the discriminator!')
        dis.disable_update()
    # Dataset
    if comm.rank == 0:
        dataset = yaml_utils.load_dataset(config)
        # # even though not new samples, use as proxy for iid validation ones.
        if hasattr(dataset, 'n_concats') and dataset.n_concats == 1:
            valid_samples = (dataset.base[:args.batch_val] + 1) * 127.5
        else:
            valid_samples = [(dataset.get_example(i)[0] + 1) * 127.5 for i in range(args.batch_val)]
            # # convert the validation to an array as required by the kl script.
            valid_samples = np.array(valid_samples, dtype=np.float32)
    else:
        _ = yaml_utils.load_dataset(config)  # Dummy, for adding path to the dataset module
        dataset = None
    dataset = chainermn.scatter_dataset(dataset, comm)
    # Iterator
    multiprocessing.set_start_method('forkserver')
    if args.multiprocessing:
        # # In minoas this might fail with the forkserver.py error.
        iterator = chainer.iterators.MultiprocessIterator(dataset, config.batchsize,
                                                          n_processes=args.loaderjob)
    else:
        iterator = chainer.iterators.SerialIterator(dataset, config.batchsize)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': iterator,
        'optimizer': opts,
        'device': device,
        'mma': mma,
    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    if not args.test:
        mainf = '{}_{}'.format(strftime('%Y_%m_%d__%H_%M_%S'), args.label)
        out = os.path.join(args.results_dir, mainf, '')
    else:
        out = 'results/test'
    if comm.rank == 0:
        create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ['loss_dis', 'loss_gen', 'kl', 'ndb', 'JS', 'dis_real', 'dis_fake']

    if comm.rank == 0:
        # Set up logging
#         trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))
        for m in models.values():
            trainer.extend(extensions.snapshot_object(
                m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))

        trainer.extend(extensions.LogReport(trigger=(config.display_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
        trainer.extend(extensions.ProgressBar(update_interval=config.display_interval))

        if gen.n_classes == 0:
            trainer.extend(sample_generate(mma.avg_model, out),
                           trigger=(config.evaluation_interval, 'iteration'),
                           priority=extension.PRIORITY_WRITER)
            print('unconditional image generation extension added.')
        else:
            trainer.extend(sample_generate_conditional(mma.avg_model, out, n_classes=gen.n_classes),
                           trigger=(config.evaluation_interval, 'iteration'),
                           priority=extension.PRIORITY_WRITER)

        trainer.extend(divergence_trainer(gen, valid_samples, metric=['kl', 'ndb'], batch=args.batch_val),
                       trigger=(config.evaluation_interval, 'iteration'),
                       priority=extension.PRIORITY_WRITER)

    # # convenience function for linearshift in optimizer:
    func_opt_shift = lambda optim1: extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                                           (config.iteration_decay_start, 
                                                            config.iteration), optim1)
    # # define the actual extensions (for optimizer shift).
    trainer.extend(func_opt_shift(opt_gen))
    trainer.extend(func_opt_shift(opt_dis))

    if args.resume:
        print("Resume Trainer")
        chainer.serializers.load_npz(args.resume, trainer)

    m1 = 'Generator params: {}. Discriminator params: {}.'
    print(m1.format(gen.count_params(), dis.count_params()))
    # Run the training
    print("start training")
    trainer.run()
    print('The output dir was {}.'.format(out))
    plot_losses_log(out, savefig=True)


if __name__ == '__main__':
    main()
