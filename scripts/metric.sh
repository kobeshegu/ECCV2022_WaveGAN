#!/usr/bin/env bash

python main_metric.py --gpu 0 --dataset flower \
--name results/flower_lofgan \
--real_dir datasets/for_fid/flower --ckpt gen_00100000.pt \
--fake_dir test_for_fid