#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --output=./reports/svm_%A_%a.out
#SBATCH --error=./reports/svm_%A_%a.err

# Check if SCRATCH is set, if not set it to /tmp
if [[ -z "${SCRATCH}" ]]; then
  SCRATCH=/tmp
else
  ml load python/3.9.0
fi

project_name=${1}
bound=${2}
num_particles=4
rsamp_crit=ess
rsamp_type=${3}

lr=${4}
tilt_type=${5}
tilt_lr=${6}
tilt_batch_size=64
tilt_hdims=64
inner_steps=100
seed=${7}

run_name=${bound}_${tilt_type}_rtype_${rsamp_type}_lr_${lr}_tlr_${tilt_lr}_${seed}

src_dir=../..

source ${src_dir}/.env/bin/activate
python3 ${src_dir}/svm_train.py \
  --bound=${bound} \
  --train_num_particles=${num_particles} \
  --rsamp_crit=${rsamp_crit} \
  --rsamp_type=${rsamp_type} \
  --eval_num_particles=2048 \
  --eval_batch_size=1 \
  --num_train_steps=10000000 \
  --lr=${lr} \
  --init_scale=0.01 \
  --min_scale_diag=1e-12 \
  --model_inner_steps=${inner_steps} \
  --tilt_inner_steps=${inner_steps} \
  --tilt_type=${tilt_type} \
  --tilt_batch_size=${tilt_batch_size} \
  --tilt_lr=${tilt_lr} \
  --dre_tilt_rnn_hdims=${tilt_hdims} \
  --dre_tilt_mlp_hdims=128 \
  --summarize_every=20000 \
  --expensive_summary_mult=10 \
  --use_wandb=True \
  --wandb_proj=${project_name} \
  --wandb_run=${run_name}  \
  --checkpoint_dir=$SCRATCH/${project_name}/${run_name} \
  --checkpoint_every=10000 \
  --checkpoints_to_keep=10 \
  --parallelism=2 \
  --seed=${seed}
