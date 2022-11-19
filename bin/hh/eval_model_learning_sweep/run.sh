#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=./reports/hh_%A_%a.out
#SBATCH --error=./reports/hh_%A_%a.err

if [[ -z "${SCRATCH}" ]]; then
  SCRATCH=/tmp
else
  ml load python/3.9.0
fi

project_name_to_eval=$1
project_name="eval_${project_name_to_eval}"
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

tilt_type=${10}
tilt_lr=${11}
tilt_lr_steps=${12}
tilt_lr_mult=${13}

bwd_tilt_mlp_hdims=32,32
bwd_tilt_rnn_hdims=32
run_name=${bound}_prop_${prop_type}_tilt_${tilt_type}_iere_${i_ext}_mlr_${lr}_tlr_${tilt_lr}_plr_${prop_lr}_${lr_steps}

src_dir=../../..
source ${src_dir}/.env/bin/activate
python3 ${src_dir}/hh_eval.py \
  --bound=${bound} \
  --proposal_type=${prop_type} \
  --prop_mlp_hdims=${prop_mlp_hdims} \
  --prop_rnn_hdims=${prop_rnn_hdims} \
  --resq_type=${resq_type} \
  --tilt_type=${tilt_type} \
  --bwd_tilt_rnn_hdims=${bwd_tilt_rnn_hdims} \
  --bwd_tilt_mlp_hdims=${bwd_tilt_mlp_hdims} \
  --train_dataset_size=10_000 \
  --val_dataset_size=64 \
  --test_dataset_size=64 \
  --eval_train=False \
  --eval_val=True \
  --eval_test=True \
  --plot=False \
  --eval_batch_size=32 \
  --num_evals=5 \
  --data_seq_len=2_001 \
  --obs_subsample=50 \
  --eval_num_particles=256 \
  --seed=${seed} \
  --i_ext_rel_error_init=${i_ext} \
  --use_wandb=True \
  --wandb_proj=${project_name} \
  --wandb_run=${run_name} \
  --checkpoint_basedir="$SCRATCH/${project_name_to_eval}" \
  --store_basedir="$SCRATCH/${project_name}" \
  --run_name=${run_name}
