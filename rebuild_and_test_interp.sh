#! /bin/bash

python -m build .
pip install --force-reinstall dist/*.whl
python3 examples/matmul_interp.py