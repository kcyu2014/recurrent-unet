#!/usr/bin/env bash

python validate.py --model_path="runs/.old/runet_eythhand-h256-1-r3-w-0.4-gate3/8376/runet_eyth_hand_best_model.pkl" --steps=32 --hidden_size=256 --gate=3 --initial=1 --model=runet --config=configs/dataset/eythhand.yml --prefix=old_test