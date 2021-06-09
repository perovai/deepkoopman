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

## Opts

The trainer/model opts are not your regular dict, rather an `addict`: check out [addict](https://github.com/mewwts/addict) to have dot access:

```python
from koop.utils import Opts

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
from koop.trainer import Trainer

trainer = Trainer.debug_trainer()
batch = trainer.dev_batch()

outputs = trainer.model(batch)
```

## Misc

* data in the original code is of shape `(num_shift + 1, batch, dim)` where `num_shifts` is `number of shifts (time steps) that losses will use (maximum is len_time - 1)`
