# Task-incremental learning on CIFAR-100
A PyTorch implementation for task-incremental learning with the Split CIFAR-100 protocol.


## Installation & requirements
The current version of the code has been tested with `Python 3.5.2` on several Linux operating systems with the following versions of PyTorch and Torchvision:
* `pytorch 1.1.0`
* `torchvision 0.2.2`
 
Assuming  Python and pip are set up, the Python-packages used by this code can be installed using:
```bash
pip install -r requirements.txt
```
However, you might want to install pytorch and torchvision in a slightly different way to ensure compatability with your version of CUDA (see https://pytorch.org/).

Finally, the code in this repository itself does not need to be installed, but two scripts should be made executable:
```bash
chmod +x main_cl.py main_pretrain.py
```

## Running experiments
Use `main_cl.py` to run individual continual learning experiments. The main options for this script are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`|`CIFAR100`)
- `--tasks`: how many tasks?

To run specific methods, use the following:
- Context-dependent-Gating (XdG): `./main_cl.py --xdg --xdg-prop=0.8`
- Elastic Weight Consolidation (EWC): `./main_cl.py --ewc --lambda=5000`
- Online EWC:  `./main_cl.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic Intelligenc (SI): `./main_cl.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main_cl.py --replay=current --distill`
- Experience Replay (ER): `./main_cl.py --replay=exemplars --budget=1000`
- Averaged Gradient Episodic Memory (A-GEM): `./main_cl.py --replay=exemplars --agem --budget=1000`

For information on further options: `./main_cl.py -h`.


## On-the-fly plots during training
With this code it is possible to track progress during training with on-the-fly plots. This feature requires `visdom`.
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main_cl.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.
