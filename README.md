# SIXO: Smoothing Inference with Twisted Objectives

This repository contains code for the paper
"SIXO: Smoothing Inference with Twisted Objectives" to appear at Neurips 2022.
The paper is available as a preprint [here](http://sixo.ai).

## Repository Structure

```
datasets.py              Dataset loaders
bounds.py                Functions for computing variational lower bounds e.g. SIXO, FIVO
smc.py                   Sequential Monte Carlo implementation
diffusion_train.py       A script for training Gaussian diffusion models
svm_train.py             A script for training the stochastic volatility model on forex data
hh_train.py              A script for training the Hodgkin-Huxley model
hh_eval.py               A script for evaluating trained Hodgkin-Huxley models
hh_inf.py                A script for performing inference in the Hodgkin-Huxley model
util.py                  Utility functions

models/                  Directory containing model implementations
 ├── base.py             Model superclasses
 ├── diffusion.py        Gaussian diffusion model implementation
 ├── hh.py               Hodgkin-Huxley model implementation
 └── svm.py              Stochastic volatility model implementation

data/                    Directory containing datasets
 └── forex_data.tsv      Foreign exchange dataset used for the SVM

test/                    Unit tests

bin/                                   Directory with bash scripts for launching training sweeps
 ├── gdd/                              Gaussian diffusion training scripts
 ├── hh/                               Hodgkin-Huxley training scripts
 │   ├── model_learning_sweep          Hodgkin-Huxley model learning scripts
 │   ├── eval_model_learning_sweep     Hodgkin-Huxley model evaluation scripts
 │   └── inf_sweep                     Hodgkin-Huxley inference scripts
 └── svm/                              SVM training scripts
```

## Running the code

First, you must install the requirements with `pip install -r requirements.txt`. The main
libraries used are JAX, numpy, scipy, matplotlib, tensorflow, pytest, chex, optax, equinox,
and snax.

Then, you can run the python scripts directly, for example, the following command will train a SVM.

```
python3 svm_train.py \
  --bound=sixo \
  --train_num_particles=4 \
  --eval_num_particles=2048 \
  --eval_batch_size=1 \
  --num_train_steps=10000000 \
  --lr=2e-4 \
  --init_scale=0.01 \
  --min_scale_diag=1e-12 \
  --model_inner_steps=400 \
  --tilt_inner_steps=100 \
  --tilt_type=bwd_dre \
  --tilt_batch_size=64 \
  --tilt_lr=1e-3 \
  --dre_tilt_rnn_hdims=32 \
  --dre_tilt_mlp_hdims=128 \
  --summarize_every=20000 \
  --expensive_summary_mult=10 \
  --checkpoint_dir=/tmp/svm \
  --checkpoint_every=10000 \
  --checkpoints_to_keep=10 \
  --seed=0
```

Alternatively, you can run training using the scripts in the `bin` directory. Each directory has
two bash files, `run.sh` and `launch_sweep.sh`. Each `run.sh` scripts accept a few arguments
and launches an individual run. The `launch_sweep.sh` scripts iterate over hyperparameter
combinations and launch a series of runs on a SLURM cluster. It is possible to use the `run.sh`
scripts without a SLURM cluster by providing the proper command-line arguments locally.
For example, the following command runs the same SVM training as above:

```
cd bin/svm && sh run.sh svm_train_proj sixo multinomial 2e-4 bwd_dre 1e-3 0
```

The run scripts assume that you have created a `venv` called `.env` in the root directory that
includes all dependencies in `requirements.txt`.

## Running the tests

To run the tests, make sure you have `pytest` installed and then run `pytest` in the main
directory.

## Citation

```
@inproceedings{lawson2022sixo,
  title={{SIXO}: Smoothing Inference with Twisted Objectives},
  author={Dieterich Lawson and Allan Ravent{\'o}s and Andrew Warrington and Scott Linderman},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=bDyLgfvZ0qJ}
}
```
