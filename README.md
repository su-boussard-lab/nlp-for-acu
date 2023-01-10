# ACU with Natural Language Processing
***
## Name
Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes

## Description
This is the implementation for acute care use prediction based on clinical notes with TF-IDF and Transformer models for the paper [Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes](https://arxiv.org/pdf/2209.13860.pdf)

## Abstract
Clinical notes are an essential component of the health record. This paper evaluates how natural language processing (NLP) can be used to identify risk of acute care use (ACU) in oncology patients, once chemotherapy starts. Risk prediction using structured health data (SHD) is now standard, but predictions using free-text formats are complex. This paper explores the use of free-text notes for prediction of ACU in leu of SHD. Deep Learning models were compared to manually engineered language features. Results show that SHD models minimally outperform NLP models; an `1-penalised logistic regression with SHD achieved a C-statistic of 0.748 (95%-CI: 0.735, 0.762), while the same model with language features achieved 0.730 (95%-CI: 0.717, 0.745) and a transformer-based model achieved 0.702 (95%-CI: 0.688, 0.717). This paper shows how language models can be used in clinical applications and underlines how risk bias is different for diverse patient groups, even using only free-text data.
## Cite Us

```
@article{fanconi2022acu_nlp,
    title={Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes}, 
    author={Claudio Fanconi and Marieke van Buchem and Tina Hernandez-Boussard},
    year={2022},
    booktitle={AMIA 2023 Informatics Summit},
}
```

## Installation
Clone the current repository
```
git clone https://code.stanford.edu/fanconic/nlp-for-acu
cd nlp-for-acu
```

We suggest to create a virtual environment and install the required packages.
```
conda create -n acu_nlp
conda activate acu_nlp
conda install -r requirements.txt
```

### Source Code Directory Tree
```
.
└── src                 # Source code for metrics and plots           
    └── utils               # Useful functions, such as loggers and config

```


## Running the Experiments
To run the models, you first need to prepare the data. For this experiment we expect four CSV files: `feature_matrix.csv` shall contain the features, `labels.csv` should contain the labels. Both of these should be indexed by a patient deidentifier number. `test_ids.csv` and `train_ids.csv` are CSV files that contain the patiend deid files of the test and training set, respectively. You can change the paths in `config.yml` file. In this file you can also set which model should be fitted, by setting their flags to either True or False

To create the TF-IDF features for the logistic regression model, run
```
python ./create_tfidf_features.py
```

To fit the models, and create predictions of the test set, run 
```
python ./fit_LASSO_models.py
```

To fit the BERT models, please use the instructions on the other [GitLab Repository](https://code.stanford.edu/boussard-lab/claudio-master-thesis).

To run the experiments and compare the models on their predictive peformance (metrics, calibration, net benefit), run
```
python ./test_model_predictions.py
```

To test the sensitivity of predictive uncertainty, run
```
python ./test_sensitivity.py
```

## Authors
- Claudio Fanconi (fanconic@ethz.ch) (cide)
- Marieke van Buchem
- Tina Hernandez-Boussard
