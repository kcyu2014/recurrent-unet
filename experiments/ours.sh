#!/usr/bin/env bash
{%- set params = [

{ "model": "runet", "dataset": "epflhand-new", "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 3, "level": 4, "step": 3, "batch_size": 4},

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
{%- set batch_size = p["batch_size"] %}
{%- set feature_scale = p["feature_scale"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --hidden_size={{ hidden * level }} \
        --scale_weight={{ scale }} \
        --unet_level={{ level }} \
        --step={{ step }} \
        --batch_size={{ batch_size }} \
        --feature_scale={{ feature_scale }} \
        --lr=1e-2 \
        --prefix=benchmark \
    > logs/cvpr/benchmark-{{ model }}-{{ dataset }}-init{{ initial }}-scale-{{ scale }}-hidden-{{ hidden*level }}-gate-{{ gate }}-level-{{ level }}-batchsize-{{batch_size}}.log \
    2>&1 &

{%- endfor %}

#--prefix=cvprablation

#{ "model": "vanillaRNNunet_NoParamShare", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 0, "level": 4, "step": 3, "batch_size": 6},
#{ "model": "vanillaRNNunet_NoParamShare", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 3, "batch_size": 6},
#{ "model": "vanillaRNNunet_NoParamShare", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 1},
#{ "model": "vanillaRNNunet_NoParamShare", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 6},
#{ "model": "vanillaRNNunet_NoParamShare", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3, "level": 4, "step": 3, "batch_size": 3},

# TODO R-UNET with only hidden as ablation.
# { "model": "gruunet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 8, "gate": 3, "gpu": 2, "level": 4, "step": 6, "batch_size": 1},
# { "model": "runethidden", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3, "level": 4, "step": 3},

# cv27 : 20000 and 20641. for multiple recurrent step, done!
# DOING: r2-gate3 doing on GPU 1. GPU0 for queing
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 0, "level": 4, "step": 4},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 0, "level": 3, "step": 4},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 0, "level": 3, "step": 2},
# DONE: r1-gate2/3


# done
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 1, "level": 2},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 1},
#{ "model": "rcnn2", "dataset": "road", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 4, "level": 1},
#{ "model": "runet", "dataset": "road", "initial": "1", "scale_weight": 0.2, "hidden": 32, "gate": 3, "gpu": 1, "level": 4},
#TODO { "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 1, "level": 2},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 1},
#
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 128, "gate": 3, "gpu": 3, "level": 3},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 128, "gate": 2, "gpu": 3, "level": 3},

# "model": "rcnn2", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 3 },
# {"model": "rcnn2", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 3 },
# {"model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 5 },
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 5 },
#{ "model": "runet", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 0 },

#CUDA_VISIBLE_DEVICES=1 python train_hand.py --config=configs/dataset/eythhand.yml --model=dru --gate=3 --initial=1 \
#--hidden_size=128 --scale_weight=0.4 --feature_scale=4 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation-softmax
#
#CUDA_VISIBLE_DEVICES=1 python train_hand.py --config=configs/dataset/eythhand.yml --model=dru --gate=3 --initial=1 \
#--hidden_size=128 --scale_weight=0.4 --feature_scale=4 --lr=5e-3 --batch_size=8 --step=12 --structure=dru --prefix=iccvablation-softmax
#
#CUDA_VISIBLE_DEVICES=2 python train_hand.py --config=configs/dataset/eythhand.yml --model=dru --gate=3 --initial=1 \
#--hidden_size=128 --scale_weight=0.4 --feature_scale=4 --lr=5e-3 --batch_size=8 --step=9 --structure=dru --prefix=iccvablation-softmax
#
#CUDA_VISIBLE_DEVICES=3 python train_hand.py --config=configs/dataset/eythhand.yml --model=dru --gate=3 --initial=1 \
#--hidden_size=128 --scale_weight=0.4 --feature_scale=4 --lr=5e-3 --batch_size=8 --step=6 --structure=dru --prefix=iccvablation-softmax

#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=dru --gate=3 --initial=1 \
#--hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=16 --step=3 --structure=dru --prefix=iccvablation

#CUDA_VISIBLE_DEVICES=2 python train_hand.py --config=configs/dataset/eythhand.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#CUDA_VISIBLE_DEVICES=2 python train_hand.py --config=configs/dataset/eythhand.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#CUDA_VISIBLE_DEVICES=2 python train_hand.py --config=configs/dataset/hofhand.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/hofhand.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/gteahand.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/egohand.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=2 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/egohand.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=2 --step=3 --structure=dru --prefix=
#
#python3 train_hand.py --config=configs/dataset/eythhand.yml --model=druresnet50bn --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/drive.yml --model=dru --gate=3 --initial=1 --hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=4 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/drive.yml --model=reclast --gate=3 --initial=1 --hidden_size=32 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=4 --step=3 --structure=reclast --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/drive.yml --model=recmid --gate=3 --initial=1 --hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=4 --step=3 --structure=recmid --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/drive.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=4 --step=3 --structure=recmid --prefix=iccvablation
#
#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-4 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=16 --step=3 --structure=dru --prefix=iccvablation
#
#
#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=druresnet50syncedbn --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=1e-2 --batch_size=16 --step=3 --structure=dru --prefix=iccvablation
#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=16 --step=3 --structure=dru --prefix=iccvablation-vgg16
#
#CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/road.yml --model=druvgg16 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#CUDA_VISIBLE_DEVICES=1 python3 train_hand.py --config=configs/dataset/road.yml --model=druresnet50 --gate=3 --initial=1 --hidden_size=256 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#
#CUDA_VISIBLE_DEVICES=1 python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=dru --gate=3 --initial=1 --hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=dru --prefix=iccvablation
#CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=sru --gate=2 --initial=1 --hidden_size=512 --scale_weight=0.4 --feature_scale=1 --lr=5e-3 --batch_size=8 --step=3 --structure=sru --prefix=iccvablation
