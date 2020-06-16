{%- set params = [
{ "model": "gruunetold", "dataset": "eythhand", "initial": "0", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },

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

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/nvidia/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --model={{ model }} \
        --hidden_size={{ hidden }} \
        --scale_weight={{ scale }} \
        --lr=1e-8 \
        --prefix=baseline \
    > logs/baseline/{{ dataset }}-{{ model }}-init{{ initial }}-scale-{{ scale }}-hidden-{{ hidden }}-gate-{{ gate }}-gpu-{{ gpu }}.log \
    2>&1 &

{%- endfor %}
# TODO fix the GRuunet.


#{ "model": "gruunetr", "dataset": "eythhand",  "gpu": 1 },
#{ "model": "unethidden", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1 },
# Too large
#{ "model": "vanillarnnunetr", "dataset": "eythhand", "initial": "1", "scale_weight": 1, "hidden": 32, "gate": 3, "gpu": 3 },
#{ "model": "vanillarnnunetr", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
#
# { "model": "gruunetr", "dataset": "hofhand",  "gpu": 2 },
#{ "model": "vanillarnnunetr", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 1 },
#{ "model": "vanillarnnunetr", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 1 },
#{ "model": "unethidden", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
#{ "model": "unethidden", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },

# FInd the reason why gruunet is so good. Because of the hidden size is set to 256.
# TODO in the DOING process.
#{ "model": "gruunetnew", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
#{ "model": "gruunetnew", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
#{ "model": "gruunetnew", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 5 },
#{ "model": "gruunetnew", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 4 },
#

# TODO next day.
#{ "model": "gruunetnew", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 5 },
#{ "model": "gruunetnew", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
