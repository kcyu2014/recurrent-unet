#!/usr/bin/env bash
{%- set params = [
{ "model": "icnet", "dataset": "road", "gpu": 1, "batch_size": 8, "loss": "multi_scale_cross_entropy"},
]
%}
{%- for p in params %}
{%- set gpu = p["gpu"] %}
{%- set model = p["model"] %}
{%- set dataset = p["dataset"] %}
{%- set batch_size = p["batch_size"] %}
{%- set loss = p["loss"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --batch_size={{ batch_size }} \
        --prefix=baseline \
        --lr=1e-9 \
        --loss={{loss}} \
    > logs/baseline/{{ model }}-{{ dataset }}-gpu{{ gpu }}-`hostname`.log \
    2>&1 &

{%- endfor %}
#{ "model": "icnet", "dataset": "hofhand", "gpu": 2, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "dataset": "epflhand-new", "gpu": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "dataset": "gteahand", "gpu": 0, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "dataset": "eythhand", "gpu": 0, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "dataset": "egohand", "gpu": 2, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "dataset": "road", "gpu": 2, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "deeplabv3", "dataset": "drive", "gpu": 2, "batch_size": 4, "loss": "cross_entropy"},
#{ "model": "deeplabv3", "dataset": "gteahand", "gpu": 2, "batch_size": 4, "loss": "cross_entropy"},
#{ "model": "deeplabv3", "dataset": "eythhand", "gpu": 1, "batch_size": 4, "loss": "cross_entropy"},

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