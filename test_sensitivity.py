""" This file contains the testing script for creating the various plots for the experiments with quantified uncertainty
Author:
    Claudio Fanconi
"""
from cProfile import label
import os
import pickle
import pandas as pd
import numpy as np
from src.utils.config import config
from src.plots import (
    plot_group_sensitivity,
    plot_group_uncertainty_binary,
    plot_word_importance,
)


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    # Load relevant test data
    feature_matrix = (
        pd.read_csv(config.data.data_path, low_memory=False)
        .sort_values(by="PAT_DEID")
        .set_index("PAT_DEID")
    )
    outcomes = pd.read_csv(config.data.label_path).set_index("PAT_DEID")
    labels_all = outcomes[config.data.label_type].reindex(feature_matrix.index)
    test_ids = pd.read_csv(config.data.test_ids)["PAT_DEID"]

    # Drop the patients that do not have notes
    feature_matrix = feature_matrix.dropna()
    labels_all = labels_all.loc[
        labels_all.index.intersection(feature_matrix.index).intersection(test_ids)
    ]
    X_test = feature_matrix.loc[labels_all.index]

    predictions = np.load(
        os.path.join(
            config.data.save_predictions, config.sensitivity.model_predictions_path
        ),
        allow_pickle=True,
    )["arr_0"]

    # Test Race
    names_dict = {
        "DEMO_White": "White",
        "DEMO_Black": "Black",
        "DEMO_Asian": "Asian",
        "DEMO_RACE_OTHER_UNK": "Other/Unknown",
    }
    plot_group_sensitivity(
        predictions=predictions,
        feature_matrix=X_test,
        names_dict=names_dict,
        group_name="Race",
        save_path=config.data.figures_path,
    )

    # Test Ethnicity
    names_dict = {
        "DEMO_Hispanic/Latino": "Hispanic/Latino",
        "DEMO_Non-Hispanic": "Non-Hispanic",
    }
    plot_group_sensitivity(
        predictions=predictions,
        feature_matrix=X_test,
        names_dict=names_dict,
        group_name="Ethnicity",
        save_path=config.data.figures_path,
    )

    # Test Gender
    plot_group_uncertainty_binary(
        predictions=predictions,
        feature_matrix=X_test,
        col_group_name="DEMO_GENDER_F",
        group_name="Gender",
        binary_dict={1: "Female", 0: "Male"},
        save_path=config.data.figures_path,
    )

    # Test Depressed
    plot_group_uncertainty_binary(
        predictions=predictions,
        feature_matrix=X_test,
        col_group_name="DEMO_DEPRESSED",
        group_name="Depressed",
        binary_dict={1: "Yes", 0: "No"},
        save_path=config.data.figures_path,
    )

    # Test Insurance Type
    names_dict = {
        "DEMO_Medicaid": "Medicaid",
        "DEMO_Medicare": "Medicare",
        "DEMO_Private": "Private",
        "DEMO_INSURANCE_OTHER_UNK": "Other & Unknown",
    }
    X_test["DEMO_INSURANCE_OTHER_UNK"] = (
        X_test["DEMO_INSURANCE_OTHER"] | X_test["DEMO_INSURANCE_UNK"]
    )
    plot_group_sensitivity(
        predictions=predictions,
        feature_matrix=X_test,
        names_dict=names_dict,
        group_name="Insurance",
        save_path=config.data.figures_path,
    )

    # Test Cancer Type
    names_dict = {
        "DEMO_breast": "Breast",
        "DEMO_gastrointestinal": "Gastrointestinal",
        "DEMO_genitourinary": "Genitourinary",
        "DEMO_gynecologic": "Gynecologic",
        "DEMO_head_neck": "Head/neck",
        "DEMO_hematopoietic_lymph": "Hematopoietic Lymph",
        "DEMO_hepatobiliary_pancreas": "Hepatobiliary Pancreas",
        "DEMO_lung_thoracic": "Lung thoracic",
        "DEMO_neurologic": "Neurologic",
        "DEMO_prostate": "Prostate",
        "DEMO_sarcoma": "Sarcoma",
    }
    plot_group_sensitivity(
        predictions=predictions,
        feature_matrix=X_test,
        names_dict=names_dict,
        group_name="Cancer Type",
        save_path=config.data.figures_path,
    )

    # Test Cancer Stage
    names_dict = {
        "DEMO_STAGE_1": "Stage 1",
        "DEMO_STAGE_2": "Stage 2",
        "DEMO_STAGE_3": "Stage 3",
        "DEMO_STAGE_4": "Stage 4",
        "DEMO_STAGE_UNK": "Unknown",
    }
    plot_group_sensitivity(
        predictions=predictions,
        feature_matrix=X_test,
        names_dict=names_dict,
        group_name="Cancer Stage",
        save_path=config.data.figures_path,
        order=True,
    )

    with open(
        os.path.join(config.data.model_path, config.sensitivity.model_path), "rb"
    ) as f:
        model = pickle.load(f)
    label_type = config.sensitivity.label_type

    filter = [c for c in X_test.columns if "WORD" in c]
    plot_word_importance(
        model=model,
        data_columns=X_test[filter].columns,
        save_path=config.data.figures_path,
        label_type=label_type,
        n=10,
    )


if __name__ == "__main__":
    main(random_state=config.seed)
