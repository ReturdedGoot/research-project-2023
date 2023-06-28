import copy
import itertools
import os
import logging
import argparse
from pathlib import Path
import random

logging.basicConfig(
     level=logging.INFO, 
    #  format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    #  datefmt='%H:%M:%S'
 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=int, default=0, help="starting id when running experiments")
    parser.add_argument('--partition', type=int, default=1, help="partition configs in this amount")

    args = parser.parse_args()
    print(args)

    softmaxwindows = ["expanding", "sliding", "full"]
    partitions = ["column", "balanced", "shuffled"]
    local_epochs = [1, 2, 5, 10]
    local_bs_sizes = [40, 64, 128]
    fracs = [1.0, 0.8, 0.5]
    # partitions = ["balanced"]
    changing_params = [softmaxwindows, partitions, local_epochs, local_bs_sizes, fracs]
    params = {
        'alg': 'WEIT',
        'dataset': 'cifar100',
        'num_classes': 100,
        'model': 'LeNet',
        'num_users': 5,
        'round': 10,
        'frac': 1.0,
        'local_bs': 40,
        'optim': 'Adam',
        'lr': 0.001,
        'lr_decay': 1e-4,
        'task': 10,
        'epoch': 100,
        'local_ep': 2,
        'labelType': 'finecoarse',
        'seed': 42
    }
    all_configs = list(itertools.product(*changing_params))

    # We are checking if this specific seed in not yet processed.

    log_path = Path('./log')
    processed_seeds = []
    for p in log_path.iterdir():
        if len(p.name.split('_')) > 1:
            files = [x.name for x in p.iterdir()]
            if 'round_data.csv' in files:
                # print(files)
                processed_seeds.append(int(p.name.split('_')[-2].split('-')[1]))
            # print(len(p.name.split('_')), p.name.split('_')[-2].split('-')[1])
    processed_seeds = list(set(processed_seeds))
    print(len(processed_seeds))
    exp_ids = list(range(args.set, len(all_configs), args.partition))
    random.shuffle(exp_ids)
    iter = 1
    for idx in exp_ids:
        if idx in processed_seeds:
            continue
        # else:
        #     print(f'[{iter}] Prcessing exp {idx}')
        #     iter += 1
        # continue
        smw, partition, local_epoch, local_bs, frac = all_configs[idx]
        # print(smw, partition)
        paramset = copy.deepcopy(params)
        paramset['softmaxtype'] = smw
        paramset['partition'] = partition
        paramset['local_ep'] = local_epoch
        paramset['local_bs'] = local_bs
        paramset['frac'] = frac
        paramset['seed'] = idx
        # Give additional name to log files
        paramset['exp_name'] = f'softmax-{smw}_partition-{partition}_le-{local_epoch}_lbs-{local_bs}_frac-{frac}_seed-{idx}'
        param_str = ('--{}={} '* len(paramset)).format(*(x for kv in paramset.items() for x in kv))
        cmd = f'python -m FedKNOW.single.main_WEIT {param_str}'
        logging.info(cmd)
        os.system(cmd)

    logging.info('>>> DONE <<<')
    logging.info('>>> Exiting now <<<')