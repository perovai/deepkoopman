# deepkoopman

Reference repo: https://github.com/BethanyL/DeepKoopman

## Install

Installing (assumes Python 3.9.2)

```bash
# create virtual env
$ python -m venv dkenv

# install dependencies
$ pip install -r requirements-3.9.2.txt

# Before pushing: check code-quality -> isort black flake8
$ pre-commit run --all-files
```

Code style: use `black` to format the code and `flake8` to lint (and help you!)

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
KeyError: 'losses'
```

## Run

Try with dummy data (for now!)

```bash
(dkenv) $ python train.py task=discrete
```

## Misc

* data in the original code is of shape `(num_shift + 1, batch, dim)` where `num_shifts` is `number of shifts (time steps) that losses will use (maximum is len_time - 1)`