# AutoML for creating hybrid Earth science models
## Abstract
<p>
Due to the availability of large sets of satellite data, an increasing number of Earth system science problems are tackled by applying machine learning.
In general, two types of methods are used for Earth system science problems:
"data-driven" methods and "theory-driven" methods. 
Data-driven methods involve the use of a large training dataset to train a machine learning model. In the context of remote sensing tasks, a machine learning model is trained by using a large set of "in situ" training data (ground truth measurements) coupled with satellite observations, where the satellite observations provide the input features and the in situ training dataset contains the target values to predict. 
However, 
in many scenarios the amount of available in situ data is limited. 
Theory-driven methods rely on the use of existing domain knowledge instead of large sets of training data. 
An example of such a method is the use of simulation models to create simulated training data.
On the downside, these models typically require extensive domain knowledge to tune correctly.
</p>
<p>
A novel perspective on data science aims to combine these data-driven and theory-driven methods: "theory-guided" data science. 
In this thesis, we introduce a theory-guided framework that incorporates both simulation models and available in situ data within a modelling pipeline. For this framework, we create an extension to the existing automated machine learning framework of Auto-sklearn. We compare the performance of this new framework to several commonly used data-driven baselines including Random forest, Multilayer perceptron, Gaussian process regression and vanilla Auto-sklearn. To facilitate this comparison, we introduce a benchmark dataset consisting of four distinct Earth system science tasks with 
preprocessed, ready-to-use in situ, simulation and remote sensing data for each task. 
From our experiments with this benchmark dataset, we conclude that for one task (leaf area index estimation), the theory-guided framework outperforms all baselines. In this task, the proposed method improves on vanilla Auto-sklearn by an increase in R<sup>2</sup> of 0.01 to 0.02 for training sizes of up to 250 in situ samples. For other tasks, vanilla Auto-sklearn consistently ranks as the best model.
</p>

## Data
<p>
In this project we composed a benchmark dataset of preprocessed, ready-to-use in situ data, satellite data and simulation data. This dataset is available on <a href="https://github.com/victorneuteboom/master-thesis-public-data">GitHub</a>.

This benchmark combines data from the following sources:
</p><ul>
  <li><a href="https://land.copernicus.eu/global/gbov/products/">Ground-Based Observations for Validation (GBOV) of Copernicus Global Land Products - Reference Measurements</a></li>
  <li><a href="https://www.slu.se/en/Collaborative-Centres-and-Projects/the-swedish-national-forest-inventory/listor/sample-plot-data/">Swedish National Forest Inventory - Sample plot data 2007-2020</a></li>
  <li><a href="https://doi.org/10.1594/PANGAEA.862886"> Valente, A et al. (2016): A compilation of global bio-optical in situ
data for ocean-colour satellite applications.</a></li>
  <li><a href="https://github.com/ESA-PhiLab/WorldCrops/tree/main/data/cropdata/Bavaria">ESA Philab WorldCrops - Bavaria Yield</a></li>
  <li><a href="https://developers.google.com/earth-engine/datasets/catalog/sentinel">Google Earth Engine Data Catalog - Sentinel Collections</a></li>
  <li><a href="https://github.com/jgomezdans/prosail">PROSAIL (Python library)</a></li>
  <li><a href="https://doi.org/10.5281/zenodo.4782707">HYDROPT: a Python framework for fast inverse 
                   modelling of multi- and hyperspectral ocean color
                   data</a></li>
</ul>
<p></p>

## Overview
This repository contains a pip-installable package in folder tgess.
The core components of the code framework are located in tgess/src. 
These components are split into two types: Data Engineering (located in tgess/src/data_engineering) and Data Science (located in tgess/src/data_science).

### Data Engineering
The Data Engineering folder contains all modules required to create and preprocess datasets. Any configurable parameters, such as file paths and names are contained in /config and automatically loaded into any module. Most functions automatically create intermediate data files, especially for long computations.

### Data Science
The Data Science folder contains all modules required to run experiments. Machine learning models, experimental setup and plotting functions are defined here. Similarly to the Data Engineering components, this folder contains a /config folder to define configurable parameters and automatically loads them. By default, results are saved in tgess/src/data_science/results.

### Usage
To recreate experiments from the master thesis, run the bash scripts located in tgess/src.
