#!/usr/bin/env bash

# Change this to your own python path
#export PATH='/cvlabdata1/home/kyu/anaconda2/envs/pytorch1.0/bin/python:$PATH'
#export UGE_LOG='/opt/uge/tmp/kyu/log'
#export UGE_ERR='/opt/uge/tmp/kyu/err'
#export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
## change working directory
#cd ~/.pycharm/pytorch-semseg
##alias python=/cvlabdata1/home/kyu/anaconda2/envs/pytorch1.0/bin/python
#. $1
#

{%- set params = [

{ "model": "unet", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32,  "gate": 2, "gpu": 1, "level": 1, "step": 3},

]
%}
{%- for p in params %}
{%- set gpu = p["gpu"] %}
{%- set model = p["model"] %}
{%- set dataset = p["dataset"] %}
{%- set initial = p["initial"] %}
{%- set scale = p["scale_weight"] %}
{%- set hidden = p["hidden"] %}
{%- set gate = p["gate"] %}
{%- set level = p["level"] %}
{%- set step = p["step"] %}

cd /cvlabdata2/home/kyu/.pycharm/hand-fix/
CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu /cvlabdata2/home/kyu/miniconda3/bin/python train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --model={{ model }} \
        --hidden_size={{ hidden * level }} \
        --scale_weight={{ scale }} \
        --unet_level={{ level }} \
        --step={{ step }} \
        --prefix=cvprablation
{%- endfor %}

#{ "model": "unetbnslim", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3 },
#{ "model": "unethidden", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3},
#{ "model": "gruunetnew", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3},