{%- set params = [

{ "model": "runet", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 0, "level": 2},
{ "model": "runet", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 0, "level": 1},
{ "model": "runet", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 0, "level": 3},
{ "model": "rcnn2", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 0, "level": 3},

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

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/nvidia/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --model={{ model }} \
        --hidden_size={{ hidden * level }} \
        --scale_weight={{ scale }} \
        --unet_level={{ level }} \
        --prefix=cvpr \
    > logs/cvpr/test-{$!}-gpu{{ gpu }}-{{ model }}-{{ dataset }}-init{{ initial }}-scale-{{ scale }}-hidden-{{ hidden }}-gate-{{ gate }}-level-{{ level }}.log \
    2>&1 &

{%- endfor %}

# done
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 1, "level": 2},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 1},


#TODO { "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 64, "gate": 3, "gpu": 1, "level": 2},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1, "level": 1},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 128, "gate": 3, "gpu": 3, "level": 3},
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 128, "gate": 2, "gpu": 3, "level": 3},

# "model": "rcnn2", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 3 },
# {"model": "rcnn2", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 3 },
# {"model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 5 },
#{ "model": "runet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 5 },
#{ "model": "runet", "dataset": "egohand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 3, "gpu": 0 },
