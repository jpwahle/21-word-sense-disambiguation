#!/bin/bash
###################################
#$ -N WSD
#$ -l h_vmem=32G
#$ -pe smp 8
#$ -o wsd_experiment.$JOB_ID.out
#$ -e wsd_experiment.$JOB_ID.err
#$ -q gpu@scc195,gpu@scc196,gpu@scc197,gpu@scc198,gpu@scc199
#$ -l gpu=1
#$ -m beas

# usage sh run.sh $MODEL_NAME_OR_PATH

# make sure set $DATA_DIR in your env
# $OUTPUT_DIR will default to .

conda activate nlp
export CUDA_VISIBLE_DEVICES=$SGE_GPU

python3 main.py --model_name_or_path $1 \
                --output_dir $OUTPUT_DIR \
                --data_dir $DATA_DIR \
                --max_epochs 3 \
                --fp16 \
                --do_eval \
                --do_train \
                --do_predict \
                --max_seq_length 160 \
                --batch_size 32 \
                --learning_rate 2e-5 \
                --hidden_dropout_prob 0.2 \
                --method lmgc \
                --shuffle \
                --num_workers 8 \
                --eval_names ALL semeval2007 semeval2013 semeval2015 senseval2 senseval3
