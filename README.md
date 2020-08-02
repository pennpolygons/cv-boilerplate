# cv-boilerplate

![](docs/CVB.gif)

Open source boilerplate for computer vision research. We solve small problems related to project management, experiment management, code styling, and model training so you can get back to research!


![License](https://img.shields.io/github/license/pennpolygons/cv-boilerplate)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)


# Installation

## Standard

```
git clone https://github.com/pennpolygons/cv-boilerplate.git
cd cv-boilerplate
make init
```

## Anaconda

```
git clone https://github.com/pennpolygons/cv-boilerplate.git
cd cv-boilerplate
make init-conda
```

# Demo

By default training uses: `/research/configs/default.yaml`

Training outputs:
- Visdom (in browser): __localhost:8000__
- Output log file, images, data files, hydra logs, Visdom logs: `/outputs` 

## Example: train using default config

```
python research/train.py 
```

## Example: train using Hydra to modify default config at runtime

```
python research/train.py mode.train.max_epochs=1
```

# Building Blocks & Choices

This project is a collection of principled choices to make CV research easier. We prioritize using tools that minimize researcher engineering work, help keep the codebase _uncluttered_, and make research easy to distribute and reproduce.

- __Python Formatting__: [Black](https://black.readthedocs.io/en/stable/)
- __Configuration Management__: [Hydra](https://hydra.cc/)
- __Machine Learning Framework__: [PyTorch](https://pytorch.org/)
- __Training / Evaluation Managment__: [PyTorch Ignite](https://pytorch.org/ignite/)
- __Visualization__: [Visdom](https://github.com/facebookresearch/visdom)
- __Typed Code__: [Types!](https://docs.python.org/3/library/typing.html)
- __Version Control__: Git