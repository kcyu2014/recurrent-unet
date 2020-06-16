{%- set params = [

{ "model": "unetbnslim", "dataset": "hofhand",  "gpu": 1 },
]
%}
{%- for p in params %}
{%- set gpu = p["gpu"] %}
{%- set model = p["model"] %}
{%- set dataset = p["dataset"] %}

CUDA_VISIBLE_DEVICES={{ gpu }} nohup python \
    train_hand.py \
        --config=configs/nvidia/{{ dataset }}.yml \
        --model={{ model }} \
        --prefix=baseline \
        --lr=5e-9 \
        --loss=cross_entropy \
    > logs/baseline/kyu-handseg-{{ model }}-{{ dataset }}-gpu{{ gpu }}.log \
    2>&1 &

{%- endfor %}

# Too large
#
# TODO in Nov 3. Test on HOF and other baseline, for unet-slim. since the unet full performance is too scary.
#{ "model": "unetgnslim", "dataset": "gteahand",  "gpu": 3 },
#{ "model": "unetbnslim", "dataset": "gteahand",  "gpu": 2 },
#{ "model": "unetgnslim", "dataset": "hofhand",  "gpu": 4 },
#{ "model": "unetbnslim", "dataset": "hofhand",  "gpu": 1 },
#{ "model": "unetgnslim", "dataset": "egohand",  "gpu": 3 },
#{ "model": "unet", "dataset": "egohand",  "gpu": 2 },
#{ "model": "unet", "dataset": "gteahand",  "gpu": 5 },