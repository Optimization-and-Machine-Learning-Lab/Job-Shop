# On the study of Curriculum Learning for inferring dispatching policies on the Job Shop Scheduling

This is an implementation of the paper "On the study of Curriculum Learning for inferring
dispatching policies on the Job Shop Scheduling". The model generates generates solutions for JSP task using Reinforced Adaptive Staircase Curriculum Learning (RASCL) strategy for training. All dependencies are mentioned in the requirements.txt.


## Commands to prepare the project

conda config --append channels conda-forge

conda create -n jssp python=3.9.7

conda activate jssp

pip install --upgrade pip

git clone THIS

pip install -r requirements.txt 

## Useful information

script file is needed to run any .py file on a cluster. 

This project uses Taillard's and DMU datasets for evaluation and randomly generated instances for training purposes.

To reproduce plots from the paper run get_plots.ipynb

To train model using certain CL strategy uncomment corresponding lines in main_train.py

To evaluate model using certain Selection strategy or certain dataset uncomment corresponding lines in main_test.py

## Useful commands

### Navigate project

conda activate jssp

cd Job-Shop

### Training
python main_train.py

### Evaluation
python main_test.py


## Useful links

### Taillard's instances
http://optimizizer.com/TA.php

### DMU instances
http://optimizizer.com/DMU.php
