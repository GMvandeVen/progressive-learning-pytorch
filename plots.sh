#!/usr/bin/env bash

./compare_hyperParams.py --pre-convE --convE-ltag=s100N --iters=5000 --seed=11
./compare.py --pre-convE --convE-ltag=s100N --iters=5000 --lambda=1000 --o-lambda=1000 --c=100 --n-seeds=4

./compare_hyperParams.py --pre-convE --convE-ltag=s100N --iters=500 --max-samples=50 --seed=11
./compare.py --pre-convE --convE-ltag=s100N --iters=500 --max-samples=50 --lambda=10000 --o-lambda=10000 --c=100 --n-seeds=4

./compare_hyperParams.py --iters=5000 --seed=11
./compare.py --iters=5000 --lambda=10000 --o-lambda=1000 --c=100 --n-seeds=4

./compare_hyperParams.py --iters=500 --max-samples=50 --seed=11
./compare.py --iters=500 --max-samples=50 --lambda=10000 --o-lambda=10000 --c=10000 --n-seeds=4

#./compare_both_pre.py --iters=5000 --lambda=10000 --lambda-pre=1000 --o-lambda=1000 --c=100 --n-seeds=4
#./compare_both_pre.py --iters=500 --max-samples=50 --lambda=10000 --o-lambda=10000 --c-pre=100 --c=10000 --n-seeds=4

./compare_both.py --lambda=10000 --lambda-500=10000 --o-lambda=1000 --o-lambda-500=10000 --c=100 --c-500=10000 --n-seeds=4
