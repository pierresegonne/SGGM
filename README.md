<div align="center">    
 
# Scalable Geometrical Generative Models | SGGM

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://pierresegonne.github.io/MScThesis/)
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/pierresegonne/SGGM/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->
</div>

An honest attempt at making uncertainty prediction with neural networks reliable and principled.

## Description

__Repository Organisation__

The repository is organized as a package, the code being located under `sggm/`.

The package information is described in the `setup.py` file. The overall settings are regrouped in the `setup.cfg` file, which mainly contains the testing ([pytest](https://docs.pytest.org/en/stable/)) and automatic linting configurations ([flake8](https://flake8.pycqa.org/en/latest/)). The `dependencies/` folder holds the built [geoml](https://bitbucket.org/soren_hauberg/geoml/src/master/) package, as it is currently not publicly released on a package management platform such as _pip_.

A simple test, `tests/test_toy.py`, ran after each commit on `master` through GitHub actions provides a sanity check on the code. It mainly flags high-level issues, such as dependency issues.

The files `run.sh` and `verify_run.py` are used for executing training jobs on a cluster, specifically a cluster using the `bsub` queue system, such as DTU's HPC ([link #1](https://www.hpc.dtu.dk/?page_id=2534), [link #2](https://itswiki.compute.dtu.dk/index.php/GPU_Cluster)), and their usage will be detailed later.

## The Code

The project relies on [Pytorch-Lightning](https://www.pytorchlightning.ai/) for handling all the nitty picky engineering details.
The execution logic is located in the `experiment.py` file while the modelling logic is located in the respective `*_model.py` file.
The experiment file can be fed parameters either directly through the CLI or through a config file (see `configs/*.yml` for inspiration) combining the custom arguments defined in `experiment.py` or specific to the model, as defined in `definitions.py`, with those predefined for the [Pytorch-Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api).

Using a config file for specifying the run parameters makes it very straightforward to use:

`python experiment.py --experiments_config configs/config_file.yml`

And is the recommended way to proceed.

## Running on a Cluster

Once you're happy with your config file, running a complete experiment accelerated on gpu is completely automatic. If you have access to a computing cluster that uses the `bsub` queue system, simply specify the correct config file in the `run.sh` executable and submit it. The `verify_run.py` simply allows to verify the experiment name and run names specified in a given config file. It is recommended to use it before submitting jobs to verify that the config file provided is valid.

### Analysis

The `analysis/` folder holds a variety of analysis scripts, which are supposed to be freely updated for the given task at hand.

* `analysis/run.py`: main entry point to extract the analysis metrics and generate the plots specific to each dataset.
* `analysis/run_ood.py`: Run analysis plots on inputs coming from a different dataset than the one used for training.
* `analysis/run_uci.py`: Shortcut to run the analysis of all uci experiments at once.
* `analysis/run_clustering.py`: Run clustering on the latent encodings of the test dataset in a VAE setting.
* `analysis/compare.py`: Generates a comparison csv for the analysis metrics for several experiments at once.

### Refitting the Encoder

`run_name` becomes `run_name_refit_encoder_$other_experiment_name`


### Citation

The Paper is currently WIP - Full reference to be provided soon!
<!-- 
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->
