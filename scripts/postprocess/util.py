import os
import sys
from itertools import product

import numpy as np
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
from experiments import util as exp_util


# public
def get_results(args, in_dir, logger=None):
    """
    Retrieve results for the multiple methods.
    """

    if logger:
        logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.method, args.use_leaf, args.update_set,
                                         args.kernel, args.target, args.lmbd, args.n_epoch, args.trunc_frac,
                                         args.check_every, args.global_op, args.local_op, args.similarity]))

    visited = set()
    results = []

    for items in tqdm(experiment_settings):

        method, use_leaf, update_set, kernel, target, lmbd, n_epoch,\
            trunc_frac, check_every, global_op, local_op, similarity = items

        template = {'method': method,
                    'use_leaf': use_leaf,
                    'update_set': update_set,
                    'kernel': kernel,
                    'target': target,
                    'lmbd': lmbd,
                    'similarity': similarity,
                    'n_epoch': n_epoch,
                    'trunc_frac': trunc_frac,
                    'check_every': check_every,
                    'random_state': args.random_state,
                    'n_jobs': args.n_jobs,
                    'global_op': global_op,
                    'local_op': local_op,
                    'similarity': similarity}

        _, hash_str = exp_util.explainer_params_to_dict(method, template)

        exp_dir = os.path.join(in_dir,
                               args.dataset,
                               args.tree_type,
                               f'rs_{args.random_state}',
                               args.inf_obj,
                               f'{method}_{hash_str}')

        method_id = f'{method}_{hash_str}'

        # skip empty experiments
        if not os.path.exists(exp_dir) or method_id in visited:
            continue

        # add results to result dict
        else:
            visited.add(method_id)

            result = _get_result(template, exp_dir)
            if result is not None:
                results.append((method_id, result))

    return results


def get_plot_dicts():
    """
    Return dict for color, line, and labels for each method.
    """
    color = {'random_': 'blue', 'minority_': 'pink', 'target_': 'cyan', 'loss_': 'yellow'}
    color['boostin_a13c8d352d437d05a9ea0fa682414bd0'] = 'orange'
    color['boostin_9e7293ec2e335fc18664e45dc2434f0c'] = 'orange'
    color['boostin_089c2ebe4906b715923c1ccf354f6bf7'] = 'orange'
    color['boostin_e88a59815ab34d3ffb6cafb4e51af75e'] = 'gray'
    color['boostin_c4fa9c6f9e90416578f695d6cd7d9ddf'] = 'gray'
    color['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'green'
    color['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'green'
    color['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'green'
    color['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'red'
    color['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'red'
    color['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'magenta'
    color['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'brown'
    color['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'brown'
    color['similarity_da2995ca8d4801840027a5128211b2d0'] = 'purple'

    line = {'random_': '-', 'minority_': '-', 'target_': '-', 'loss_': '-'}
    line['boostin_a13c8d352d437d05a9ea0fa682414bd0'] = '-'
    line['boostin_9e7293ec2e335fc18664e45dc2434f0c'] = '--'
    line['boostin_089c2ebe4906b715923c1ccf354f6bf7'] = ':'
    line['boostin_e88a59815ab34d3ffb6cafb4e51af75e'] = '--'
    line['boostin_c4fa9c6f9e90416578f695d6cd7d9ddf'] = ':'
    line['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = '-'
    line['trex_c026a1d65c79084fe50ec2a8524b2533'] = '--'
    line['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = ':'
    line['loo_590f53e8699817c6fa498cc11a4cbe63'] = '-'
    line['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = '--'
    line['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = '-'
    line['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = '-'
    line['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = '--'
    line['similarity_da2995ca8d4801840027a5128211b2d0'] = '-'

    label = {'random_': 'Random', 'minority_': 'Minority', 'target_': 'Target', 'loss_': 'Loss'}
    label['boostin_a13c8d352d437d05a9ea0fa682414bd0'] = 'BoostIn'
    label['boostin_9e7293ec2e335fc18664e45dc2434f0c'] = 'BoostIn_SGN'
    label['boostin_089c2ebe4906b715923c1ccf354f6bf7'] = 'BoostIn_SIM'
    label['boostin_e88a59815ab34d3ffb6cafb4e51af75e'] = 'BoostIn_NTG'
    label['boostin_c4fa9c6f9e90416578f695d6cd7d9ddf'] = 'BoostIn_HESS'
    label['trex_0e3f576fe95f9fdbc089be2b13e26f89'] = 'TREX'
    label['trex_c026a1d65c79084fe50ec2a8524b2533'] = 'TREX_exp'
    label['trex_f6f04e6ea39b41fecb05f72fc45c1da8'] = 'TREX_alpha'
    label['loo_590f53e8699817c6fa498cc11a4cbe63'] = 'LOO'
    label['loo_cd26d9e10ce691cc69aa2b90dcebbdac'] = 'LOO_exp'
    label['dshap_9c4e142336c11ea7e595a1a66a7571eb'] = 'DShap'
    label['leaf_influence_6bb61e3b7bce0931da574d19d1d82c88'] = 'LeafInf'
    label['leaf_influence_cfcd208495d565ef66e7dff9f98764da'] = 'LeafInf_SP'
    label['similarity_da2995ca8d4801840027a5128211b2d0'] = 'Sim.'

    return color, line, label


# private
def _get_result(template, in_dir):
    """
    Obtain the results for this baseline method.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result
