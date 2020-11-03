#!/bin/bash
###################################
#$ -N GilBERT
#$ -l h_vmem=32G
#$ -pe smp 8
#$ -o GilBERT_AdamW_bs.$JOB_ID.out
#$ -e GilBERT_AdamW_bs.$JOB_ID.err
#$ -q gpu@scc195,gpu@scc196,gpu@scc197,gpu@scc198,gpu@scc199
#$ -l gpu=1
#$ -M jan.philip.wahle@gmail.com
#$ -m beas

conda activate nlp
export CUDA_VISIBLE_DEVICES=$SGE_GPU

python3 main.py --model_name_or_path $1 \
                --output_dir /Users/jp/Desktop/results \
                --data_dir /Users/jp/projects/data/semcor_csv \
                --max_epochs 3 \
                --fp16 \
                --do_eval \
                --do_train \
                --do_predict \
                --max_seq_length 160 \
                --batch_size 32 \
                --learning_rate 2e-5 \
                --hidden_dropout_prob 0.2 \
                --method bgp \
                --shuffle \
                --eval_names ALL semeval2007 semeval2013 semeval2015 senseval2 senseval3
