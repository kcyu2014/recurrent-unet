{%- set params = [



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
{%- set f_scale = p["scale"] %}
{%- set lr = p["lr"] %}



CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/dataset/{{ dataset }}.yml \
        --model={{ model }} \
        --gate={{ gate }}  \
        --initial={{ initial }} \
        --model={{ model }} \
        --hidden_size={{ hidden * level }} \
        --scale_weight={{ scale }} \
        --unet_level={{ level }} \
        --step=3 \
        --feature_scale={{ f_scale }} \
        --lr=1e-{{ lr }} \
        --prefix=cvprablation \
    > logs/cvpr/ablation-gpu{{ gpu }}-{{ model }}-{{ dataset }}-init{{ initial }}-scale-{{ scale }}-hidden-{{ hidden*level }}-gate-{{ gate }}-level-{{ level }}.log \
    2>&1 &

{%- endfor %}

# TODO Doing this, full feature scale test here .
#{ "model": "vanillarnnunet", "dataset": "road", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 5, "level": 4, "scale": 1, "lr": 3},
#{ "model": "runet", "dataset": "road", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 7, "level": 4, "scale": 1, "lr": 3},