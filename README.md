# deepkoopman

Reference repo: https://github.com/BethanyL/DeepKoopman

## Install

Installing (assumes Python 3.9.2)

```
# create virtual env
python -m venv dkenv

# install dependencies
pip install -r requirements-3.9.2.txt
```

## Run

Try with dummy data (for now!)

```
(dkenv) $ python train.py
```

## Misc

* data in the original code is of shape `(num_shift + 1, batch, dim)` where `num_shifts` is `number of shifts (time steps) that losses will use (maximum is len_time - 1)`