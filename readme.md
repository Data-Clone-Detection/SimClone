#Repo For SIMCLONE

# 1.build classifier
## 1.1 clone injection
`data/data_process.py` will inject clone and call 1.2 to generate feature
## 1.2 feature generation 
`data/feature_utils.py` contain all function for feature generation 
## 1.3 training/testing
`train.py` train and evaluate SimClone and LTC

# 2.visualization
the visualization phase can be seen in `visualization.py`, using jupyter notebook might provide a better visualization result