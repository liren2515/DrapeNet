# DrapeNet: Garment Generation and Self-Supervised Draping
working on it.


Environment:
* Ubuntu 20.04
* python 3.8.6
* PyTorch 1.13.1 w/ CUDA 11.7

## Setup:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
pip install numpy open3d einops hesiod fvcore tensorboard trimesh cython networkx
```

Install `torch` and `pytorch3d`:
```
pip install https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```

Build and install `meshudf`:
```
cd meshudf
source setup.sh
```

## Encoder-Decoder:
```
cd encdec
python3 preprocess_udf.py </path/to/meshes> </out/path>
python3 train_encdec.py  # add </path/to/run/cfg/file> to restore training
python3 export_codes.py </path/to/run/cfg/file>
python3 export_meshes.py </path/to/run/cfg/file>
```

## Editing:
```
cd editing
python3 create_dset.py
python3 compute_weights.py
python3 edit.py
```