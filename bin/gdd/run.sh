#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G


if [[ -z "${SCRATCH}" ]]; then
  SCRATCH=/tmp
else
  ml load python/3.9.0
fi


PROJECT_NAME=$1
BOUND=$2
RGRAD_MODE=$3
TILT_TRAINING_METH=$4
LEARN_P=$5
LEARN_Q=$6
LEARN_R=$7
SEED=$8
RUN_NAME=${BOUND}_tilt_${TILT_TRAINING_METH}_rgrad_${RGRAD_MODE}_p_${LEARN_P}_q_${LEARN_Q}_r_${LEARN_R}_${SEED}

SRC_DIR=$HOME/dev/fivo

PARAMS="--bound=${BOUND}
        --drift=1
        --resampling_gradient_mode=${RGRAD_MODE}
        --tilt_train_method=${TILT_TRAINING_METH}
        --syn_data_num_series=1000
        --syn_data_num_timesteps=10
        --train_num_particles=10
        --eval_num_particles=128
        --train_steps=500000
        --train_inner_steps=1000
        --tilt_inner_steps=1000
        --lr=1e-3
        --use_wandb
        --tilt_lr=1e-3
        --batch_size=32
        --tilt_batch_size=64
        --tilt_mlp_hdims=32
        --summarize_every=10000
        --wandb_proj=${PROJECT_NAME}
        --checkpoint_dir=$SCRATCH/${PROJECT_NAME}/${RUN_NAME}
        --seed=${SEED}
       "

if [ "${LEARN_P}" = 1 ]; then
  PARAMS+=" --learn_p"
fi
if [ "${LEARN_Q}" = 1 ]; then
  PARAMS+=" --learn_q"
fi
if [ "${LEARN_R}" = 1 ]; then
  PARAMS+=" --learn_r"
fi

echo "Run name: ${RUN_NAME}"
echo "Params: ${PARAMS}"

source $SRC_DIR/.env/bin/activate
python3 $SRC_DIR/diffusion_train.py ${PARAMS}
