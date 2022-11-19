import sys
sys.path.append("../sixo")
import hh_train
import tempfile
import pytest


def default_config():
  return hh_train.parser.parse_args([])


@pytest.mark.parametrize("prop_type", ["bootstrap", "filtering", "smoothing"])
@pytest.mark.parametrize("bound", ["elbo", "iwae", "fivo"])
def test_train_fivo(prop_type, bound):
  with tempfile.TemporaryDirectory() as dirpath:
    cfg = default_config()
    cfg.bound = bound
    cfg.proposal_type = prop_type
    cfg.tilt_type = 'none'
    cfg.lr = 1e-4
    cfg.train_dataset_size = 4
    cfg.val_dataset_size = 4
    cfg.test_dataset_size = 2
    cfg.data_seq_len = 10
    cfg.obs_subsample = 2
    cfg.num_train_steps = 1
    cfg.train_num_particles = 2
    cfg.eval_num_particles = 2
    cfg.tilt_inner_steps = 1
    cfg.model_inner_steps = 1
    cfg.summarize_every = cfg.tilt_inner_steps + cfg.model_inner_steps
    cfg.expensive_summary_mult = 1
    cfg.checkpoint_dir = dirpath
    cfg.checkpoints_to_keep = 1
    cfg.use_wandb = False
    cfg.parallelism = 1
    hh_train.train_hh(cfg)


@pytest.mark.parametrize("prop_type", ["bootstrap", "smoothing"])
def test_train_sixo(prop_type):
  with tempfile.TemporaryDirectory() as dirpath:
    cfg = default_config()
    cfg.bound = 'sixo'
    cfg.proposal_type = prop_type
    cfg.tilt_type = 'bwd_dre'
    cfg.lr = 1e-4
    cfg.train_dataset_size = 4
    cfg.val_dataset_size = 4
    cfg.test_dataset_size = 2
    cfg.data_seq_len = 10
    cfg.obs_subsample = 2
    cfg.num_train_steps = 1
    cfg.train_num_particles = 2
    cfg.eval_num_particles = 4
    cfg.tilt_inner_steps = 1
    cfg.model_inner_steps = 1
    cfg.summarize_every = cfg.tilt_inner_steps + cfg.model_inner_steps
    cfg.expensive_summary_mult = 1
    cfg.checkpoint_dir = dirpath
    cfg.checkpoints_to_keep = 1
    cfg.use_wandb = False

    hh_train.train_hh(cfg)
