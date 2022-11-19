import tempfile
import sys
sys.path.append("../sixo")
import diffusion_train
import pytest


def default_config():
  return diffusion_train.parser.parse_args([])


@pytest.mark.parametrize("bound,grad_mode,learn_p,learn_q,learn_r",
        [("sixo", "score_fn_rb", True, True, True),
         ("sixo", "none", True, True, True),
         ("iwae", "none", True, True, False),
         ("fivo", "score_fn_rb", True, True, False),
         ("fivo", "none", True, True, False),
         ("sixo", "none", True, True, False)])
def test_train_one_step(bound, grad_mode, learn_p, learn_q, learn_r):

  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.syn_data_num_series = 10
    config.syn_data_num_timesteps = 10
    config.train_num_particles = 4
    config.train_steps = 1
    config.lr = 1e-4
    config.batch_size = 2
    config.eval_num_particles = 4
    config.checkpoint_dir = dirpath
    config.drift = 0.
    config.bound = bound
    config.resampling_gradient_mode = grad_mode
    config.learn_p = learn_p
    config.learn_q = learn_q
    config.learn_r = learn_r
    if bound == "sixo":
      config.tilt_train_method = "unified"
    diffusion_train.main(config)


@pytest.mark.parametrize("grad_mode", ["score_fn_rb", "none"])
@pytest.mark.parametrize("learn_p,learn_q,learn_r",
        [(True, True, True),
         (False, True, True),
         (False, False, True)])
def test_train_one_step_dre(grad_mode, learn_p, learn_q, learn_r):

  with tempfile.TemporaryDirectory() as dirpath:
    config = default_config()
    config.syn_data_num_series = 10
    config.syn_data_num_timesteps = 10
    config.train_num_particles = 4
    config.train_steps = 1
    config.train_inner_steps = 1
    config.tilt_inner_steps = 1
    config.lr = 1e-4
    config.tilt_lr = 1e-4
    config.batch_size = 2
    config.tilt_batch_size = 2
    config.eval_num_particles = 4
    config.checkpoint_dir = dirpath
    config.drift = 0.
    config.bound = "sixo"
    config.tilt_training_method = 'dre'
    config.resampling_gradient_mode = grad_mode
    config.learn_p = learn_p
    config.learn_q = learn_q
    config.learn_r = learn_r

    diffusion_train.main(config)
