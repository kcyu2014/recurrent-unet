#!/usr/bin/env bash
# Done in Nov. 1st. 131
{ "model": "unetgn", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 6 },
{ "model": "unetbn", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 3 },
{ "model": "gruunetr", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1 },
{ "model": "gruunet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
{ "model": "vanillarnnunet", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 2, "gpu": 4 },
{ "model": "vanillarnnunetr", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 2, "gpu": 5 },

# Nov 1. 131

# Nov 1. 142
{ "model": "unetgn", "dataset": "gteahand",  "gpu": 1 },
{ "model": "unetbn", "dataset": "gteahand",  "gpu": 3 },

# Nov.1 K8s
{ "model": "gruunetr", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 1 },
{ "model": "vanillarnnunet", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 2, "gpu": 4 },
{ "model": "vanillarnnunetr", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 2, "gpu": 5 },

# TODO (in kubernetes)
# Feature scale=2
{ "model": "gruunet", "dataset": "gteahand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 3, "gpu": 2 },
{ "model": "runet", "dataset": "epflhand", "initial": "1", "scale_weight": 0.4, "hidden": 256, "gate": 2, "gpu": 5 },

# Note that, the baseline is broken possibly due to the clip gradient.

# TODO
{ "model": "rcnn2", "dataset": "hofhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 5 },
{ "model": "rcnn2", "dataset": "eythhand", "initial": "1", "scale_weight": 0.4, "hidden": 32, "gate": 2, "gpu": 3 },
