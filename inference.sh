#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path llava_merged_model \
    --question-file ./playground/data_infer/questions.jsonl \
    --image-folder ./playground/data_infer/images/ \
    --answers-file ./playground/data_infer/finetuned_llava_answers.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
