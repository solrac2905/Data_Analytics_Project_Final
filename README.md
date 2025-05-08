# What is this for?

This folder should be used to store configuration files used by Kedro or by separate tools.

This file can be used to provide users with instructions for how to reproduce local configuration with their own credentials. You can edit the file however you like, but you may wish to retain the information below and add your own section in the section titled **Instructions**.

## Local configuration

The `local` folder should be used for configuration that is either user-specific (e.g. IDE configuration) or protected (e.g. security keys).

> *Note:* Please do not check in any local configuration to version control.

## Base configuration

The `base` folder is for shared configuration, such as non-sensitive and project-related configuration that may be shared across team members.

WARNING: Please do not put access credentials in the base configuration folder.

## Find out more
You can find out more about configuration from the [user guide documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html).

-------------------------

Codes are divided into pipeline and non-pipeline framework. There are 4 files in non-pipeline framework located in notebooks folder:

- EDA_FINAL.ipynb (Exploratory Data Analytics)
- PCA_FINAL.ipynb
- Preprocessing_and_Models_FINAL.ipynb
- Sensitivity_Analysis_and_Feature_Importance_FINAL.ipynb

## If you want to run notebooks folder (non-pipeline), go to the Kedro project folder and run the following command (select the kedro kernel in jupyter notebook to run it):

```bash
kedro jupyter notebook
```

## It will open a new tab in your browser where you can execute all the notebooks that you want.

-------------------------

## Recommendations to run our pipeline code:

## In order to run the code, first, please install all the requirements in the requirements.txt file.

## How to install


To install them, run:

```
pip install -r requirements.txt
```

## In order to execute code by console, you can use:

## If you want to run the entire project:

```bash
kedro run
```

## If you want to run a specific node (example with the logistic node):

```bash
kedro run --nodes=logistic_regression_node
```

### Name of the nodes:
- preprocessing_raw_node  
- apply_log_transform_node  
- scale_node  
- train_test_split_node  
- feature_selection_node  
- apply_sampling_smote_rus_node  
- logistic_regression_node  
- decision_tree_node  
- random_forest_node  
- xgboost_node  
- neural_network_node  
- catboost_node  
- stacked_classifier_node  
- results_node  

## If you want to run a specific pipeline (example with the data science pipeline):

```bash
kedro run --pipeline=data_science
```

### Name of the pipelines:
- data_processing  
- data_science  

