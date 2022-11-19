#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3G
#SBATCH --output=./reports/hh_%A_%a.out
#SBATCH --error=./reports/hh_%A_%a.err

if [[ -z "${SCRATCH}" ]]; then
  SCRATCH=/tmp
else
  ml load python/3.9.0
fi

project_name=$1
bound=$2
i_ext=$3
seed=$4

lr=$5
lr_steps=$6
lr_mult=$7
prop_type=$8
prop_lr=$9

resq_type=full_sg
prop_mlp_hdims=32,32
prop_rnn_hdims=32

model_inner_steps=400
tilt_inner_steps=100
total_steps=500

tilt_type=${10}
tilt_lr=${11}
tilt_lr_steps=${12}
tilt_lr_mult=${13}

bwd_tilt_mlp_hdims=32,32
bwd_tilt_rnn_hdims=32
run_name=${bound}_prop_${prop_type}_tilt_${tilt_type}_iere_${i_ext}_mlr_${lr}_tlr_${tilt_lr}_plr_${prop_lr}_${lr_steps}

src_dir=../../..
source ${src_dir}/.env/bin/activate
python3 ${src_dir}/hh_train.py \
  --bound=${bound} \
  --num_train_steps=10_000_000 \
  --train_batch_size=4 \
  --proposal_type=${prop_type} \
  --prop_lr=${prop_lr} \
  --prop_mlp_hdims=${prop_mlp_hdims} \
  --prop_rnn_hdims=${prop_rnn_hdims} \
  --resq_type=${resq_type} \
  --tilt_type=${tilt_type} \
  --tilt_batch_size=32 \
  --bwd_tilt_rnn_hdims=${bwd_tilt_rnn_hdims} \
  --bwd_tilt_mlp_hdims=${bwd_tilt_mlp_hdims} \
  --model_inner_steps=${model_inner_steps} \
  --tilt_inner_steps=${tilt_inner_steps} \
  --checkpoint_every=${total_steps} \
  --summarize_every=${total_steps} \
  --expensive_summary_mult=10 \
  --train_dataset_size=10_000 \
  --val_dataset_size=64 \
  --data_seq_len=2_001 \
  --obs_subsample=50 \
  --train_num_particles=4 \
  --eval_num_particles=256 \
  --lr=${lr} \
  --lr_steps=${lr_steps} \
  --lr_mult=${lr_mult} \
  --tilt_lr=${tilt_lr} \
  --tilt_lr_steps=${tilt_lr_steps} \
  --tilt_lr_mult=${tilt_lr_mult} \
  --seed=${seed} \
  --parallelism=1 \
  --p_grad_max_norm=-1 \
  --i_ext_rel_error_init ${i_ext} \
  --use_wandb=True \
  --wandb_proj=${project_name} \
  --wandb_run=${run_name} \
  --checkpoint_dir="$SCRATCH/${project_name}/${run_name}"
