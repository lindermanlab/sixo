import tempfile
import sys
sys.path.append("../sixo")
import svm_train
import pytest


def default_config():
  return svm_train.parser.parse_args([])


@pytest.mark.parametrize("bound", ["fivo", "vsmc"])
def test_train_forex_for_one_step_no_aux(bound):
  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.lr = 1e-4
    config.num_train_steps = 1
    config.train_num_particles = 2
    config.eval_num_particles = 4
    config.num_bound_evals = 1
    config.eval_batch_size = 1
    config.summarize_every = 1
    config.checkpoint_dir = dirpath
    config.checkpoints_to_keep = 1
    config.bound = bound
    svm_train.train_svm(config)


def test_train_dre_tilt():
  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.bound = 'sixo'
    config.tilt_type = 'bwd_dre'
    config.tilt_train_set_size = 4
    config.tilt_batch_size = 2
    config.lr = 1e-4
    config.num_train_steps = 2
    config.model_inner_steps = 1
    config.tilt_inner_steps = 1
    config.train_num_particles = 2
    config.eval_num_particles = 4
    config.num_bound_evals = 1
    config.eval_batch_size = 1
    config.summarize_every = 100
    config.checkpoint_every = 100
    config.checkpoint_dir = dirpath
    svm_train.train_svm(config)


@pytest.mark.parametrize("window", [1, 3])
def test_quadrature_tilt(window):
  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.bound = 'sixo'
    config.tilt_type = 'quadrature'
    config.tilt_window = window
    config.quad_tilt_deg = 3
    config.lr = 1e-4
    config.num_train_steps = 2
    config.train_num_particles = 2
    config.eval_num_particles = 4
    config.summarize_every = 1
    config.checkpoint_every = 1
    config.checkpoint_dir = dirpath
    config.checkpoints_to_keep = 1
    svm_train.train_svm(config)


def test_checkpointing(capsys):
  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.bound = 'sixo'
    config.tilt_type = 'bwd_dre'
    config.tilt_train_set_size = 4
    config.tilt_batch_size = 2
    config.lr = 1e-4
    config.num_train_steps = 2
    config.model_inner_steps = 1
    config.tilt_inner_steps = 1
    config.train_num_particles = 2
    config.eval_num_particles = 4
    config.summarize_every = 100
    config.checkpoint_every = 2
    config.checkpoint_dir = dirpath
    config.checkpoints_to_keep = 1
    config.seed = 0
    svm_train.train_svm(config)
    svm_train.train_svm(config)
    assert "Loaded checkpoint at step 2 from" in capsys.readouterr().out
