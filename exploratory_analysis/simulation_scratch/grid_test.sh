#!/bin/bash
cd "$(dirname $0)"
DEF="1e-4 3.162e-4 1e-3 3.162e-3 1e-2 3.162e-2 1e-1 3.162e-1 1"
NOTINY="1e-3 3.162e-3 1e-2 3.162e-2 1e-1 3.162e-1 1"
for seed in 1 2; do
	../venv/bin/python -W ignore sbc_sampler.py --n_genes 6000 --p 200 --seed $seed --split --n_iters 6000 --burn_in 1000 --model_grid $DEF --cutoff 1e-3
	../venv/bin/python -W ignore sbc_sampler.py --n_genes 6000 --p 200 --seed $seed --split --n_iters 6000 --burn_in 1000 --model_grid $NOTINY --cutoff 5e-3
done
