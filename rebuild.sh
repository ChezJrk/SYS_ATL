#! /bin/bash

python -m build .
pip install --force-reinstall dist/*.whl