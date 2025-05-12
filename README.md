# Point Set Diffusion


<!-- A one line description of the project -->
This is the reference implementation of our ICLR 2025 paper [Unlocking Point Processes through Point Set Diffusion][paper].

</div>

## Citation
If you build upon this work, please cite our paper as follows:
```
@inproceedings{luedke2025psdiff,
    title={Unlocking Point Processes through Point Set Diffusion},
    author={David L{\"u}dke and Enric Rabasseda Ravent{\'o}s and Marcel Kollovieh and Stephan G{\"u}nnemann},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/pdf?id=4anfpHj0wf}
}
```

## Getting started
<!-- This section summarizes the basic requirements and the installation process to properly run and reproduce the code -->

### Basic requirements
<!-- List of basic requirements needed to properly run the code -->
The code has been tested on a cluster of Linux nodes using [SLURM][slurm-site].<br>
We _cannot guarantee_ the functioning of the code if the following requirements are _not_ met:


### Installation
<!-- List the steps needed to properly install and run the code -->
> To properly install and run our code we recommend using a virtual environment (e.g., created via [`pyenv-virtualenv`][pyenv-virtualenv-site] or [`conda`][conda-site]).

The entire installation process consists of 3 steps. You can skip step 0 at you own "risk".

#### (_Optional_) Step 0: create a virtual environment
In the following we show how to create the environment via [`pyenv`][pyenv-site] and [`pyenv-virtualenv`][pyenv-virtualenv-site].
The steps are the following:
- install [`pyenv`][pyenv-site] (if you don't have it yet) by following the [original guidelines][pyenv-install-site];
- install the correct Python version:
    ```sh
    pyenv install 3.10.4
    ```
- create a virtual environment with the correct version of Python:
    ```sh
    pyenv virtualenv 3.10.4 ps-diff
    ```

#### Step 1: clone the repository, change into it and (_optional_) activate the environment
This step allows you to download the code in your machine, move into the correct directory and (_optional_) activate the correct environment.
The steps are the following:
- clone the repository:
    ```sh
    git clone https://github.com/davecasp/ps-diff.git
    ```
- change into the repository:
    ```sh
    cd point-set-diff
    ```
- (_optional_) activate the environment
    ```sh
    pyenv activate ps-diff
    ```

#### Step 2: install the code as a local package
All the required packages are defined in the `pyproject.toml` file and can be easily installed via [`pip`][pip-site] as following:
```sh
pip install -r requirements.txt
```

## Run code

Configuring experiments and running the code is done via hydra. If you are unfamiliar with how hydra works please check out the [documentation](https://hydra.cc/docs/intro/).

### Train model on paper configs

To run our Model for a datatype:
```sh
./train.py -m --config-name config_name
```
where ```config_name``` should be ```spp_train``` or ```stpp_train``` or ```tpp_train```. All seeds and datasets are scheduled as a gridsearch via the multirun flag.

### Condional modelling

Example will be added in due time

<!-- Python & libraries websites -->
[python-site]: https://www.python.org
[pytorch-site]: https://pytorch.org
[pytorch-install-site]: https://pytorch.org/get-started/locally/
[pyg-site]: https://pytorch-geometric.readthedocs.io/en/latest/index.html#
[pyg-install-site]: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
[lit-site]: https://www.pytorchlightning.ai
[slurm-site]: https://slurm.schedmd.com/documentation.html
[pyenv-virtualenv-site]: https://github.com/pyenv/pyenv-virtualenv
[pyenv-site]: https://github.com/pyenv/pyenv
[pyenv-install-site]: https://github.com/pyenv/pyenv#installation
[conda-site]: https://docs.conda.io/en/latest/
[pip-site]: https://pip.pypa.io/en/stable/
<!-- Internal references -->
[installation-guide-ref]: ./docs/installation.md
<!-- Other variables -->
[paper]: https://www.cs.cit.tum.de/daml/point-set-diffusion/