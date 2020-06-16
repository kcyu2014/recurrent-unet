#!/usr/bin/env bash
#{ "model": "deeplabv3", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 2, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 2, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "unet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 2, "step": 1, "batch_size": 4, "loss": "cross_entropy"},
#{ "model": "gruunet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 3, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "gruunet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 2, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "gruunet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 1, "batch_size": 4, "loss": "multi_step_cross_entropy"},

#{ "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 3, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 2, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 1, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 2, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#{ "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 1, "batch_size": 4, "loss": "multi_step_cross_entropy"},
#]
# { "model": "runethidden",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 4, "step": 1, "batch_size": 4, "loss": "multi_step_cross_entropy"},

{%- set params = [
 {"model": "runet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 4, "loss": "multi_step_cross_entropy"},
 {"model": "runet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 2, "batch_size": 4, "loss": "multi_step_cross_entropy"},
 {"model": "runet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 1, "batch_size": 4, "loss": "multi_step_cross_entropy"},
]
%}
{%- for p in params %}
{%- set gpu = p["gpu"] %}
{%- set model = p["model"] %}
{%- set dataset = p["dataset"] %}
#{%- set initial = p["initial"] %}
{%- set scale = p["scale_weight"] %}
{%- set hidden = p["hidden"] %}
{%- set gate = p["gate"] %}
#{%- set level = p["level"] %}
{%- set step = p["step"] %}
{%- set batch_size = p["batch_size"] %}
{%- set feature_scale = p["feature_scale"] %}
{%- set loss = p["loss"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python train_hand.py \
        --config=configs/dataset/road-benchmark.yml \
        --model={{ model }} \
        --gate=3  \
        --initial=1 \
        --hidden_size={{ hidden }} \
        --scale_weight={{ scale }} \
        --unet_level=4 \
        --step={{ step }} \
        --batch_size={{ batch_size }} \
        --feature_scale={{ feature_scale }} \
        --lr=1e-2 \
        --loss={{ loss }} \
        --benchmark \
        --prefix=benchmark \
    > logs/benchmark-road/{{ model }}-fs-{{ feature_scale }}-hidden-{{ hidden  }}-step{{ step }}-batchsize-{{batch_size}}.log \
    2>&1

{%- endfor %}

#{ "model": "icnet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 3, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 3, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "icnet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 3, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},

#{ "model": "unetvgg16",  "feature_scale": 1, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 1, "batch_size": 4}
# "model": "druresnet50",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 3, "level": 4, "step": 2, "batch_size": 4},
#{ "model": "druresnet50",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 3, "level": 4, "step": 2, "batch_size": 4},
#{ "model": "unetresnet50",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 1, "batch_size": 4},
#{ "model": "unetvgg16gn",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 2, "batch_size": 4},
#{ "model": "deeplabv3", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 3, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},
#{ "model": "refinenet", "feature_scale": 4,  "scale_weight": 0.4, "hidden": 64, "gpu": 3, "step": 1, "batch_size": 4, "loss": "multi_scale_cross_entropy"},

#{ "model": "druvgg16",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 2, "level": 4, "step": 1, "batch_size": 4},
#{ "model": "druvgg16",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 2, "level": 4, "step": 2, "batch_size": 4},
#
#{ "model": "gruunet",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 1},
#{ "model": "druresnet50bn",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 4},
#{ "model": "druvgg16",  "feature_scale": 4, "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2, "level": 4, "step": 3, "batch_size": 4},
