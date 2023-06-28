# Federated Continual Learning

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)

## Requirements

* Python 3.8
* `pip`

(Optionally) run in a virtual environment:

```bash
python -m venv venv
#activate enironment
source venv/bin/acticate

# Run python code
python <command> <args>

# To exit virtual enironment:
deactivate
```

Install all the packages from requirements.txt

```bash
pip install -r requirements.txt 
```

## Running experiments (updated)

```bash
python FedKNOW/run_exp_softmax.py
```

Without arguments, all the experiments will be executed one after each other in random order.

```text
usage: run_exp_softmax.py [-h] [--set SET] [--partition PARTITION]

optional arguments:
  -h, --help            show this help message and exit
  --set SET             starting id when running experiments
  --partition PARTITION
                        partition configs in this amount
```

The arguments `set` start at the id of the value of `set`
The argument `partition` splits the list of experiments in `partition` amounts.
Using this combined let you execute all experiments in parallel with multiple processes/terminal. For example:

In terminal 1: `python FedKNOW/run_exp_softmax.py --set 0 --partition 3`  
In terminal 2: `python FedKNOW/run_exp_softmax.py --set 1 --partition 3`  
In terminal 3: `python FedKNOW/run_exp_softmax.py --set 2 --partition 3`  

Each terminal will execute a 3rd of the experiments

The code will try to use `cuda` is that is available.

### Create plots

```bash
python exp_analyze.py 
```

It will create plots in the root directory based on all the data in the `log` directory

## Running single experiments (example)

1. `cd` into the root directory
2. Run the following command for 10 clients with balanced partition, 10 rounds per task, and 10 tasks per client:

```bash
python -m FedKNOW.single.main_WEIT --alg=WEIT --dataset=cifar100 --num_classes=100 --model=LeNet --num_users=5 --round=10 --frac=1.0 --local_bs=64 --optim=Adam --lr=0.001 --lr_decay=0.0001 --task=10 --epoch=100 --local_ep=10 --labelType=finecoarse --seed=318 --softmaxtype=full --partition=shuffled --exp_name=experiment-318
```

## Arguments

**Note:** The number of classes (`num_classes`) is dependent on the dataset. `Cifar100` has 100 classes when the `labelType` is `finecoarse`, and it has 20 classes otherwise.

```text
--alg: The name of the continual learning algorithm
--dataset: Dataset name thats being used
--num_classes: The total number of classes with all task datasets combined
--model: The model name.
--num_users: The total number of clients
--round: The number of rounds per task
--frac: The fraction of clients to train per round
--local_bs: The batch size to use for training a task (per-client)
--optim, --lr, --lr_decay: Model hyperparameters
--task: The number of tasks per-client
--epoch: Redundant arg, will be scrapped. For now, set it to the product of --task and --round
--local_ep: The number of training epochs per-round for each client
--partition: The dataset partition mechanism. Use "balanced", "unique", "column", or "shuffled"
--download: Additional argument needed to download the dataset if running for the first time. Can be skipped later on.
--labelType: Which label to use for the partitioning (fine label, coarse label, or define tasks with coarse label but train with the fine label (finecoarse)
--softmaxtype: The window type for the classifier output. Sliding window type for the output layer (expanding, sliding, or full)
```

## Example of Results
The effect of using different amounts of local epoch when training multiple tasks.
![Results](./img/fig_effect_local_epochs.png)

## FedKNOW

For the original repository docs, check: [https://github.com/LINC-BIT/FedKNOW/blob/main/README.md](https://github.com/LINC-BIT/FedKNOW/blob/main/README.md)
