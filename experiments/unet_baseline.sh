#!/usr/bin/env bash
{%- set params = [

{ "model": "unet", "dataset": "epflhand-new", "feature_scale": 4, "initial": "1", "gpu": 3, "batch_size": 8, "loss": "cross_entropy"},

]
%}
{%- for p in params %}
{%- set gpu = p["gpu"] %}
{%- set model = p["model"] %}
{%- set dataset = p["dataset"] %}
{%- set batch_size = p["batch_size"] %}
{%- set loss = p["loss"] %}
{%- set feature_scale = p["feature_scale"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --batch_size={{ batch_size }} \
        --prefix=baseline \
        --feature_scale={{ feature_scale }} \
        --lr=1e-2 \
        --loss={{loss}} \
    > logs/baseline/{{ model }}-{{ dataset }}-gpu{{ gpu }}.log \
    2>&1 &

{%- endfor %}

#{ "model": "unet_deep_as_dru", "dataset": "hofhand", "initial": "1", "gpu": 0, "batch_size": 6, "loss": "cross_entropy"},
#{ "model": "unet_deep_as_dru", "dataset": "eythhand", "initial": "1", "gpu": 0, "batch_size": 6, "loss": "cross_entropy"},
#{ "model": "unet_deep_as_dru", "dataset": "gteahand", "initial": "1", "gpu": 2, "batch_size": 3, "loss": "cross_entropy"},
#{ "model": "unet_deep_as_dru", "dataset": "egohand", "initial": "1", "gpu": 1, "batch_size": 2, "loss": "cross_entropy"},
#{ "model": "unet_deep_as_dru", "dataset": "epflhand-new", "initial": "1", "gpu": 3, "batch_size": 6, "loss": "cross_entropy"},

#{ "model": "unetgnslim", "dataset": "eythhand", "initial": "1", "gpu": 0, "batch_size": 6, "loss": "cross_entropy"},
#{ "model": "unetbnslim", "dataset": "eythhand", "initial": "1", "gpu": 3, "batch_size": 6, "loss": "cross_entropy"},
#{ "model": "unet", "dataset": "cityscapes", "initial": "1", "gpu": 3, "batch_size": 4, "loss": "cross_entropy"},

# Too large

#{ "model": "unet", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3 , "gpu": 2},
#{ "model": "unetbnslim", "dataset": "epflhand-new", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3 , "gpu": 2},

# TODO in Nov 3. Test on HOF and other baseline, for unet-slim. since the unet full performance is too scary.
#{ "model": "unetgnslim", "dataset": "gteahand",  "gpu": 3 },
#{ "model": "unetbnslim", "dataset": "gteahand",  "gpu": 2 },
#{ "model": "unetgnslim", "dataset": "hofhand",  "gpu": 4 },
#{ "model": "unetbnslim", "dataset": "hofhand",  "gpu": 1 },
#{ "model": "unetgnslim", "dataset": "egohand",  "gpu": 3 },
#{ "model": "unet", "dataset": "egohand",  "gpu": 2 },
#{ "model": "unet", "dataset": "gteahand",  "gpu": 5 },

#python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=unet --batch_size=16 --prefix=baseline --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=unet_expand_all --batch_size=16 --prefix=baseline --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/cityscapes.yml --model=unet_expand --batch_size=16 --prefix=baseline --feature_scale=1 --lr=1e-8 --loss=cross_entropy

python3 train_hand.py --config=configs/dataset/gteahand.yml --model=unetvgg16 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/gteahand.yml --model=unetresnet50 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

python3 train_hand.py --config=configs/dataset/hofhand.yml --model=unetvgg16 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/hofhand.yml --model=unetresnet50 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

python train_hand.py --config=configs/dataset/epflhand-new.yml --model=unetvgg16 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python train_hand.py --config=configs/dataset/epflhand-new.yml --model=unetresnet50 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

CUDA_VISIBLE_DEVICES=0 python train_hand.py --config=configs/dataset/egohand.yml --model=unetvgg16 --batch_size=2 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
CUDA_VISIBLE_DEVICES=0 python train_hand.py --config=configs/dataset/egohand.yml --model=unetresnet50 --batch_size=2 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

python3 train_hand.py --config=configs/dataset/egohand.yml --model=unetvgg16 --batch_size=2 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/egohand.yml --model=unetresnet50 --batch_size=2 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

python3 train_hand.py --config=configs/dataset/gteahand.yml --model=unetresnet50bn --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy


CUDA_VISIBLE_DEVICES=1 python train_hand.py --config=configs/dataset/drive.yml --model=unet --batch_size=4 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
CUDA_VISIBLE_DEVICES=2 python train_hand.py --config=configs/dataset/drive.yml --model=unetvgg16 --batch_size=4 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
python3 train_hand.py --config=configs/dataset/drive.yml --model=unetresnet50 --batch_size=4 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy


CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/road.yml --model=unetvgg16 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy
CUDA_VISIBLE_DEVICES=3 python3 train_hand.py --config=configs/dataset/epflhand-benchmark.yml --model=unetresnet50 --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

CUDA_VISIBLE_DEVICES=0 python3 train_hand.py --config=configs/dataset/epflhand-new.yml --model=unet --batch_size=8 --prefix=iccvablation --feature_scale=1 --lr=1e-8 --loss=cross_entropy

