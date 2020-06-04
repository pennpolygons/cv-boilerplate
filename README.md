# cv-boilerplate
Open source boilerplate for computer vision research

![License](https://img.shields.io/github/license/pennpolygons/cv-boilerplate)

# Mission

We believe that solving small problems related to project management, experiment management, code styling, and model training will benefit the community by making it easier to jump into research. 

## Building Blocks & Choices

This project is, at its essence, a collection of principled choices to make CV research easier. We prioritize using tools that minimize researcher choices, help keep the codebase _uncluttered_, and make research easy to distribute and reproduce. Note, these choices are not set in stone, and do not reflect any affiliation with any specific institution. Our choices reflect the state of tooling as it currently exists.

- __Python Formatting__: [Black](https://black.readthedocs.io/en/stable/)
- __Configuration Management__: [Hydra](https://hydra.cc/)
- __Machine Learning Framework__: [PyTorch](https://pytorch.org/)
- __Training / Evaluation Managment__: [PyTorch Ignite](https://pytorch.org/ignite/)
- __Visualization__: [Visdom](https://github.com/facebookresearch/visdom)
- __Typed Code__: [Types!](https://docs.python.org/3/library/typing.html)
- __Version Control__: Git