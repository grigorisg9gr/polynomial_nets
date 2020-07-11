import matplotlib as mpl
mpl.use('Agg')
from os import makedirs, getcwd
from os.path import basename, exists, join, isfile, dirname, isdir
import numpy as np
import shutil
from pathlib import Path
from functools import reduce
import operator
from time import strftime
import json
from textwrap import wrap
import matplotlib.pyplot as plt

import source.yaml_utils as yaml_utils


def printtime(msg, time_format='%a %d/%m %H:%M:%S'):
    """
    Wraps the typical print(msg) with a time to display
    the time.
    """
    print('[{}] {}'.format(strftime(time_format), msg))


def get_by_nested_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def get_config_value(config, path, default=None):
    """
    Fetches the value of a nested key or returns default value. 
    Wrapper around get_by_nested_path().
    """
    error = False
    try:
        value = get_by_nested_path(config, path)
    except KeyError:
        value, error = default, True
    return value, error


def create_result_dir(result_dir, config_path, config):
    if not exists(result_dir):
        makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        shutil.copy(fn, '{}/{}'.format(result_dir, basename(fn)))

    copy_to_result_dir(config_path, result_dir)
    # # define the paths we want to copy from. Some of those might
    # # not exist; copy only the existing ones.
    pool = [['models', 'generator', 'fn'], ['models', 'encoder', 'fn'],
            ['models', 'decoder', 'fn'], ['models', 'discriminator', 'fn']]
    for path1 in pool:
        val, err = get_config_value(config, path1)
        if not err:
            copy_to_result_dir(val, result_dir)
        
    copy_to_result_dir(
        config.dataset['dataset_fn'], result_dir)
    copy_to_result_dir(
        config.updater['fn'], result_dir)


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    return gen, dis


def load_models_cgan(config, rocgan=False):
    """ Load all the network models based on the yml config. """
    # # convenience function for loading a net:
    loadnet = lambda net_conf: yaml_utils.load_model(net_conf['fn'], net_conf['name'], 
                                                     net_conf['args'])
    # # load the encoder/decoder and discriminator.
    enc = loadnet(config.models['encoder'])
    dec = loadnet(config.models['decoder'])
    dis = loadnet(config.models['discriminator'])
    
    if not rocgan:
        return enc, dec, dis
    else:
        # # load the second encoder (ae pathway) in this case.
        enc_ae = loadnet(config.models['encoder_ae'])
        return enc, dec, dis, enc_ae


def ensure_config_paths(config, pb=None, verbose=True):
    """
    Parses the config (i.e. a dict) and ensures the
    paths with label 'fn' exist, or tries to replace
    it with local paths.
    """
    def _get_key(config, key):
        try:
            config[key]
            return True
        except (AttributeError, KeyError):
            return False
    # # set the base name (if the modules should change dir). If pb is provided,
    # # use that, otherwise use the current working directory.
    pbase = pb if pb is not None and isdir(pb) else getcwd()
    # # boolean to understand if something is changed.
    changed = False
    if _get_key(config, 'models'):
        # # iterate over all models and change paths.
        for key in config['models'].keys():
            m_conf = config['models'][key]
            if 'fn' in m_conf.keys() and not isfile(m_conf['fn']):
                # # modify the path (since the yml is for deepmux.)
                modelp = Path(m_conf['fn'])
                m_conf['fn'] = join(pbase, modelp.parts[-2], modelp.parts[-1])
                changed = True
    # # similarly change the dataset path.
    if _get_key(config, 'dataset'):
        dbp = Path(config['dataset']['dataset_fn'])
        if not dbp.exists():
            config['dataset']['dataset_fn'] = join(pbase, dbp.parts[-2], dbp.parts[-1])
            changed = True
    if _get_key(config, 'updater'):
        updp = Path(config['updater']['fn'])
        if not updp.exists():
            config['updater']['fn'] = join(pbase, updp.parts[-3], updp.parts[-2], updp.parts[-1])
            changed = True
    if verbose and changed:
        print('Changed the paths in config to base: {}.'.format(pbase))
    return config


def moving_average(values, n=3):
    """ Moving average in numpy (values is a list/array). """
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_losses_log(pout, savefig=True, title_len=30, name_out='losses.png', 
                    enable_ma=False, ma_n=5, interp_short=False):
    """
    Plots a host of losses (hardcoded) based on the log. The log and the
    output dir are assumed to be in pout (convenient for running time).
    ARGS:
      enable_ma: (bool, optional) If True, the moving average of the saved
          values is plotted.
      interp_short: (bool, optional) If True, the lists shorter than the log, 
          e.g. ones not exported frequently are interpolated.
    """
    # # Naming conventions: l[name] is loss of [name], vl[n] is validation
    # # loss of [n] var, bp[var] is best position of [var]. 
    printtime('Plotting losses.')
    # # convenience function if moving average is requested.
    ma = lambda x: moving_average(x, n=ma_n) if enable_ma else x
    # # convenience function to set the subtitle length to many lines.
    set_title = lambda ax, text, length=title_len: ax.set_title('\n'.join(wrap(text, length)))
    # # log1: The list with the values exported per iteration. 
    # # Each element of the list is (expected to be) a dictionary.
    log1 = json.load(open(join(pout, 'log'), 'r'))
    total_lines = len(log1)
    # # empty lists for the losses. 
    # # ldis: loss discriminator, lgadv: adversarial loss of generator, 
    ldis, lgadv, ll1  = [], [], []
    vlmssim, vlmmae, vlsssim, bpssim, vfid = [], [], [], [], []
    # # additional vars to save.
    d_fake, d_real, real_min, fake_max = [], [], [], []
    for cnt, dict_iter in enumerate(log1):
        # # plot the training losses.
        ldis.append(dict_iter['loss_dis'])
        if 'lgen_adv' in dict_iter.keys():
            lgadv.append(dict_iter['lgen_adv'])
        if 'loss_l1' in dict_iter.keys():
            ll1.append(dict_iter['loss_l1'])
        # # plot the dis_fake/real.
        d_fake.append(dict_iter['dis_fake'])
        d_real.append(dict_iter['dis_real'])
        if 'real_min' in dict_iter.keys():
            real_min.append(dict_iter['real_min'])
            fake_max.append(dict_iter['fake_max'])
        # # append the validation losses.
        if 'mssim' in dict_iter.keys():
            vlmssim.append(dict_iter['mssim'])
            vlsssim.append(dict_iter['sdssim'])
        if 'mmae' in dict_iter.keys():
            vlmmae.append(dict_iter['mmae'])
        if 'bssim' in dict_iter.keys():
            bpssim.append(dict_iter['bssim'])
        if 'FID' in dict_iter.keys():
            vfid.append(dict_iter['FID'])

    # # get the export frequency from log.
    freq = log1[1]['iteration'] - log1[0]['iteration']

    pool_to_plot = [
        (ldis, 'Discriminator adversarial loss', []),
        (lgadv, 'Generator adversarial loss', []),
        (ll1, 'Training loss L1', []),
        (d_real, 'Real vs fake (red)', d_fake),
        (vlmssim, '(Valid) Mean ssim', []),
        (vlsssim, '(Valid) Std ssim', []),
        (vlmmae, '(Valid) Mean MAE', []),
        (real_min, 'Real min vs fake max (red)', fake_max),
        (bpssim, 'Best valid (ssim) iter', []),
        (vfid, 'Frechet Inception Distance', []),
    ]
    # # define the maximum rows and columns to support. 
    max_rows, max_cols = 3, 4
    f, axarr = plt.subplots(max_rows, max_cols, sharex=True, 
                            figsize=(5 * max_cols, 7 * max_rows))
    # # c1: counter for the plots added.
    c1, total = 0, max_rows * max_cols
    # # iterate over the pool above and plot them.
    for cnt1, (list1, str1, l2) in enumerate(pool_to_plot):
        if len(list1) > 0:
            # # select the subplot.
            ax = axarr[c1 // max_cols, c1 % max_cols]
            if interp_short and len(list1) < total_lines // 4:
                # # in this case, we interpolate (staircase) the 
                # # variable that was exported infrequently.
                list1 = np.repeat(list1, int(total_lines // len(list1)))                
            ax.plot(ma(list1))
            set_title(ax, str1)
            if len(l2) > 0:
                # # in this case, add a second variable to plot.
                ax.plot(ma(l2), c='r')
            # # increase the counter for the plots made.
            c1 += 1
    if savefig:
        plt.savefig(pout + name_out, bbox_inches='tight', pad_inches=0.)
    return f


