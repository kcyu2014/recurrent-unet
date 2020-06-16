#!/usr/bin/env bash
{%- set params = [

{ "model": "gruunetnew", "dataset": "cityscapes", "feature_scale": 2, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 0, "batch_size": 4},

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
{%- set batch_size = p["batch_size"] %}
{%- set feature_scale = p["feature_scale"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --model={{ model }} \
        --hidden_size={{ hidden }} \
        --scale_weight={{ scale }} \
        --feature_scale={{ feature_scale }} \
        --lr=1e-2 \
        --batch_size={{ batch_size }} \
        --structure=gru \
        --prefix=baseline \
    > logs/baseline/{{ dataset }}-{{ model }}-init{{ initial }}-scale-{{ scale }}-hidden-{{ hidden }}-gate-{{ gate }}-gpu-{{ gpu }}.log \
    2>&1 &

{%- endfor %}

CUDA_VISIBLE_DEVICES=1,2 python3 train_hand.py --config=configs/dataset/cityscapes.yml \
--model=reclast --gate=3 --initial=1 --hidden_size=32 \
--scale_weight=0.4 --feature_scale=1 --lr=1e-2 --batch_size=16 --structure=gru --prefix=baseline

# { "model": "gruunetnew", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "batch_size": 6},
# { "model": "gruunetnew", "dataset": "cityscapes", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3, "batch_size": 2},
# { "model": "runethidden", "dataset": "cityscapes", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "batch_size": 4},
# { "model": "vanillarnnunetr", "dataset": "cityscapes", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "batch_size": 4},

#{ "model": "unethidden", "dataset": "eythhand", "initial": "0", "scale_weight": 0.4, "hidden": 128, "gate": 3,  "gpu": 1},
#{ "model": "unethidden", "dataset": "eythhand", "initial": "1", "scale_weight": 1, "hidden": 128, "gate": 3,  "gpu": 2},

# TODO DOING final cross validation.
# 130 { "model": "unethidden", "dataset": "eythhand", "initial": "1", "scale_weight": 1, "hidden": 128, "gate": 3,  "gpu": 7},

# 130
#{ "model": "unethidden", "dataset": "eythhand", "initial": "0", "scale_weight": 0.4, "hidden": 128, "gate": 3,  "gpu": 3},
#{ "model": "unethidden", "dataset": "eythhand", "initial": "1", "scale_weight": 1, "hidden": 128, "gate": 3,  "gpu": 3},

# iccvlabsrv 27. running now.
#{ "model": "unethidden", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 128, "gate": 3,  "gpu": 1},
#{ "model": "unethidden", "dataset": "eythhand", "initial": "0", "scale_weight": 1, "hidden": 128, "gate": 3,  "gpu": 2},


#{ "model": "unethidden", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3,  "gpu": 5},

# TODO DOING
#{ "model": "vanillarnnunetr", "dataset": "road",  "gpu": 1, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3},
#{ "model": "gruunetnew", "dataset": "road", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3},
# { "model": "gruunetnew", "dataset": "road", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
#

#{ "model": "gruunetr", "dataset": "hofhand",  "gpu": 1 },
# Too large
#{ "model": "vanillarnnunetr", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
# { "model": "gruunetr", "dataset": "hofhand",  "gpu": 2 },
#{ "model": "vanillarnnunetr", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 1 },
#{ "model": "vanillarnnunetr", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 1 },
#{ "model": "unethidden", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
#{ "model": "unethidden", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },

# FInd the reason why gruunet is so good. Because of the hidden size is set to 256.
# Done.
#{ "model": "gruunetnew", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
#{ "model": "gruunetnew", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
#{ "model": "gruunetnew", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 5 },
#{ "model": "gruunetnew", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 4 },
#

# Done
#{ "model": "gruunetnew", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 5 },
#{ "model": "gruunetnew", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },

python3 train_hand.py --config=configs/dataset/drive.yml --model=vanillarnnunetr --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=4 --step=3 --structure=vanillarunet --prefix=iccvablation

CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=recmid --gate=3 --initial=1 --hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=recmid --prefix=iccvablation
CUDA_VISIBLE_DEVICES=1 python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=reclast --gate=3 --initial=1 --hidden_size=32 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=recmid --prefix=iccvablation

CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/road.yml --model=reclast --gate=3 --initial=1 --hidden_size=32 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=reclast --prefix=iccvablation



