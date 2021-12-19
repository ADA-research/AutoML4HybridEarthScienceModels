# Master thesis
Still a work in progress, some parts may not work directly out-of-the-box. 

## Overview
This repository contains a pip-installable package in folder tgess.
The core components of the code framework are located in tgess/src. 
These components are split into two types: Data Engineering (located in tgess/src/data_engineering) and Data Science (located in tgess/src/data_science).

### Data Engineering
The Data Engineering folder contains all modules required to create and preprocess datasets. In pipeline.py, multiple functions are chained together to create each dataset. Any configurable parameters, such as file paths and names are contained in config.yaml and automatically loaded into any module. Most functions automatically create intermediate data files, especially for long computations.

### Data Science
The Data Science folder contains all modules required to run experiments. Machine learning models, experimental setup and plotting functions are defined here. Similarly to the Data Engineering components, this folder contains a config.yaml to define configurable parameters and automatically loads them. By default, results are saved in tgess/src/data_science/results.

### Usage
First, you have to install the pip package located in tgess, using the command below. Otherwise, some modules will have failing imports.
```
pip install -e tgess
```
Then, you have to install the requirements located in requirements.txt (NOTE: current requirements.txt is not working --> install packages manually). 
```
pip install -r requirements.txt
```

To run a Data Engineering pipeline or Data Science pipeline, you can run the following command.
```
python cli.py [pipeline_type] [pipeline name]
```

For full description of command line options, run the following command.
```
python cli.py -h
```
