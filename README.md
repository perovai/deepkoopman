# deepkoopman

Reference repo: https://github.com/BethanyL/DeepKoopman

## Install

Installing (assumes Python 3.9.2)

```bash
# create virtual env
$ python -m venv dkenv

# activate the environment
source dkenv/bin/activate

# updgrade pip
pip install --upgrade pip

# install dependencies
$ pip install -r requirements-3.9.2.txt

# Before pushing: check code-quality -> isort black flake8
$ pre-commit run --all-files
```

Code style: use `black` to format the code and `flake8` to lint (and help you!)

### Getting started with comet_ml

1. Create an account https://www.comet.ml/signup
2. In your account settings go to `Developper Information`, then `Generate an API Key` and copy it.
3. Create a `~/.comet.config` file in your home with the following lines:

  ```ini
  [comet]
  api_key=<YOUR COPIED API_KEY>
  workspace=<YOUR USERNAME>
  ```

1. Share your username with victor to be added as a collaborator and you're good to go :)

## Code structure

* The main component to execute code is a `Trainer` object. It's an object-oriented programming way of stitching together many moving parts: pytorch models, data loaders, optimizers, schedulers, losses etc.
* The trainer is parametrized by `opts` (options, =hyper-parameters) which are loaded in `train.py`
* `train.py` uses `minydra` as a command-line parser (victor made it) meaning any option can be overwritten from the command-line
* `Trainer`s have a `setup()` method to create the datasets, models, etc. then a `train()` method calling:
  * `run_epoch()` to execute an epoch
  * `run_validation()` to compute the validation loss
  * `update_early_stopping()` to stop training if the val loss has not improved for the past `opts.early.patience` epochs (7 as of now)


## Data

DeepKoopman Datasets

https://drive.google.com/file/d/1XRfa4EQ3JauAmlMxE92_kBvGg126DrfM/view?usp=sharing

unzip and put sub-folders in `datasets/`
## Opts

The trainer/model opts are not your regular dict, rather an `addict`: check out [addict](https://github.com/mewwts/addict) to have dot access:

```python
from aiphysim.utils import Opts

opts = Opts({"losses": {"recon": True, "koopman": True}})
opts.epochs = 4
opts.batch_size = 8
print(opts.losses.recon)
opts.unknown
```

prints:

```python
True
...
KeyError: 'unknown'
```

### opts.yaml

The `opts` keys can be specified:

* per task

    ```yaml
    param:
      task: value
    ```

* for all tasks
    ```yaml
    param: value
    ```

  or

    ```yaml
    param:
      key1: value1
      key2: value2
      ...
    ```

## Run

Note1: It expects csv files to be stored in `datasets/DiscreteSpectrumExample` folder.

Note2: To log data to [comet.ml](https://comet.ml) checkout <https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup> for instructions on how to setup your comet_ml API key: either in `~/.comet.config` or as an environment variable COMET_API_KEY. If you do not want to log progress to `comet`, then pass the argument `comet.use=False` in the command line.

```bash
# using minydra, all options can be overwritten from the command-line
(dkenv) $ python train.py task=discrete epochs=1
```

```python
# dev mode: quickly getting a trainer and a batch
from aiphysim.trainer import Trainer

trainer = Trainer.debug_trainer()
batch = trainer.dev_batch()

outputs = trainer.model(batch)
```

## Misc

* data in the original code is of shape `(num_shift + 1, batch, dim)` where `num_shifts` is `number of shifts (time steps) that losses will use (maximum is len_time - 1)`

## Reading perovskite data

```python
from pathlib import Path
import h5py

file_path = Path() / "datasets" / "mini-dataset-modest-volhard.h5"

f5 = h5py.File(file_path, 'r')

dataset_0 = f5["trajectory_0"]

print(dict(dataset_0.attrs))
# {'delay': 0.0, 'i': 0.2104, 'set': 'training_set1', 't3': 0.0}

array = dataset_0[()]
print(array.shape)
# (2499, 3, 6) -> 2499 time steps of a 3x3 complex matrix 
# array[0, 0, :2] is a + ib, array[0, 0, 2:4] is c + id and array[0, 0, 4:6] is e + if
```

## Physical Constraints and Background Reading

### Physical Constraints

There are a range of physical constraints / rules which dynamical open system should obey, depending on the precise setup and situation. Starting with the most basic:

* Trace of the density matrix should equal exactly 1 at all times
* The density matrix should equal its Hermitian conjugate
* As a corollary to the above - the diagonal elements should all be real between 0 and 1, and off-diagonal elements should be complex-conjugate. 

A few more specific properties which may or may not be relevant for our setup - I'm probably missing some here.

* Depending on system-bath coupling, at equilibrium, the off-diagonal elements should decay to zero
* Also depending on system-bath coupling, at equilibrium, the diagonal elements should thermalize (obey detailed balance condition) to a certain temperature.
* There are certain properties to do with state purity and information flow between system and bath, though they might not be so easily untangled over the course of a laser experiment.

### Background Reading

I've had a look at some recent papers on ML method for schemical and open quantum/many-body dynamics. Some are straightforward and some I need to read more carefully. See the list here.

* [SINDy Nets](https://www.pnas.org/content/pnas/116/45/22445.full.pdf)
* [Our Koopman pproach](https://www.nature.com/articles/s41467-018-07210-0)
* [Technical work on open quantum systems](https://arxiv.org/pdf/2009.05580.pdf)
* [Deep autoregressive approach](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.124.020503?casa_token=FQRxHr56qG4AAAAA%3AezXAl8-sx5g-qjE_BpXMunfRSRL8VTyAz-KsxTE7uT9Uq34d7kwPPZl9KyUbvSDe0HaJW8gIEuaZoek)
* [VAMPNets](https://www.nature.com/articles/s41467-017-02388-1)
* [Kindof basic-seeming approach](https://www.sciencedirect.com/science/article/pii/S0301010418304336?casa_token=f52aa7YsslYAAAAA:8XG5IfnhslZd_SF38mlnOvsuhyaOo3y7dry1ocXH1uaEbONSZGaTAP2tsBor6dT6K96KKViLWR0)
* [Nice review from the molecular kinetics point of view](https://pubs.acs.org/doi/full/10.1021/acs.chemrev.0c01195)
