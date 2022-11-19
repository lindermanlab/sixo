#!/bin/bash

if [[ -z "${SCRATCH}" ]]; then
  SCRATCH=/tmp
else
  ml load python/3.9.0
fi

project_name=${1}
bound=${2}
seed=${3}
subsample=${4}
mlp_hdims=${5}
rnn_hdims=${6}

prop_type=${7}
prop_lr=${8}
lr_steps=${9}
lr_mult=${10}

resq_type=full_sg
model_inner_steps=400
tilt_inner_steps=100
total_steps=500

tilt_type=${11}
tilt_lr=${12}
tilt_lr_steps=${13}
tilt_lr_mult=${14}
noise_scale=${15}

run_name=${bound}_prop_${prop_type}_tilt_${tilt_type}_tlr_${tilt_lr}_plr_${prop_lr}_${lr_steps}_hdim_${mlp_hdims}_ss_${subsample}_ns_${noise_scale}

src_dir=../../..
source ${src_dir}/.env/bin/activate
python3 ${src_dir}/hh_inf.py \
  --bound=${bound} \
  --num_train_steps=600_000 \
  --train_batch_size=4 \
  --proposal_type=${prop_type} \
  --prop_lr=${prop_lr} \
  --prop_mlp_hdims=${mlp_hdims} \
  --prop_rnn_hdims=${rnn_hdims} \
  --resq_type=${resq_type} \
  --tilt_type=${tilt_type} \
  --tilt_batch_size=32 \
  --bwd_tilt_rnn_hdims=${rnn_hdims} \
  --bwd_tilt_mlp_hdims=${mlp_hdims} \
  --model_inner_steps=${model_inner_steps} \
  --tilt_inner_steps=${tilt_inner_steps} \
  --checkpoint_every=${total_steps} \
  --summarize_every=${total_steps} \
  --expensive_summary_mult=50 \
  --train_dataset_size=1_000 \
  --data_seq_len=2_001 \
  --obs_subsample=${subsample} \
  --noise_scale=${noise_scale} \
  --train_num_particles=4 \
  --eval_num_particles="8,16,32,64,128,256" \
  --num_evals=10 \
  --lr_steps=${lr_steps} \
  --lr_mult=${lr_mult} \
  --tilt_lr=${tilt_lr} \
  --tilt_lr_steps=${tilt_lr_steps} \
  --tilt_lr_mult=${tilt_lr_mult} \
  --seed=${seed} \
  --parallelism=1 \
  --p_grad_max_norm=-1 \
  --use_wandb=True \
  --wandb_proj=${project_name} \
  --wandb_run=${run_name} \
  --checkpoint_dir="$SCRATCH/${project_name}/${run_name}"
