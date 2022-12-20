""" This file contains the testing script for the BLRs to calculate the predictive metrics, calibration and net benefit
Author:
    Claudio Fanconi
"""
import os
import pandas as pd
import numpy as np
from src.utils.config import config
from src.metrics import nr_variables_used, print_results
from src.plots import calibration_plot, kaplan_meyer_plot, net_benefit_plot


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    # Load relevant test data
    feature_matrix = feature_matrix = (
        pd.read_csv(config.data.data_path, low_memory=False)
        # .sort_values(by="PAT_DEID")
        .set_index("PAT_DEID")
    )
    outcomes = pd.read_csv(config.data.label_path).set_index("PAT_DEID")
    labels_all = outcomes[config.data.label_type].reindex(feature_matrix.index)
    test_ids = pd.read_csv(config.data.test_ids)["PAT_DEID"]

    first_op_df = pd.read_csv(config.data.info_df, parse_dates=[8, 9, 12])

    # Drop the patients that do not have notes
    feature_matrix = feature_matrix.dropna()
    labels_all = labels_all.loc[
        labels_all.index.intersection(feature_matrix.index).intersection(test_ids)
    ]
    X_test = feature_matrix.loc[labels_all.index]

    for label_type in config.data.label_type:
        y_test = labels_all[label_type]
        print(f"-------------------- Results for {label_type} ----------------------")
        # load the predictions / predictive distributions of the four models:
        tabular_LASSO = np.load(
            os.path.join(
                config.data.save_predictions,
                f"tabular_model_predictions_{label_type}.npz",
            ),
            allow_pickle=True,
        )["arr_0"]
        language_LASSO = np.load(
            os.path.join(
                config.data.save_predictions,
                f"language_model_predictions_{label_type}.npz",
            ),
            allow_pickle=True,
        )["arr_0"]
        fusion_LASSO = np.load(
            os.path.join(
                config.data.save_predictions,
                f"fusion_model_predictions_{label_type}.npz",
            ),
            allow_pickle=True,
        )["arr_0"]
        language_BERT = np.load(
            os.path.join(
                config.data.save_predictions,
                f"language_bert_predictions_{label_type}.npz",
            ),
            allow_pickle=True,
        )["arr_0"]
        fusion_BERT = np.load(
            os.path.join(
                config.data.save_predictions,
                f"fusion_bert_predictions_{label_type}.npz",
            ),
            allow_pickle=True,
        )["arr_0"]
        # Print predictive performance of models:
        predictions = {
            "Tabular LASSO": tabular_LASSO,
            "Language LASSO": language_LASSO,
            "Fusion LASSO": fusion_LASSO,
            "Language BERT": language_BERT,
            "Fusion BERT": fusion_BERT,
        }

        # Calculate the predictive metrics
        for name, y_pred in predictions.items():
            print(name)
            print(
                nr_variables_used(
                    feature_matrix, config.data.model_path, name, label_type
                ),
            )
            print_results(y_test, y_pred)
            # Create Kaplan-Meyer risk analysis
            kaplan_meyer_plot(
                predictions=pd.DataFrame(
                    data=np.array([y_pred, y_test]).T,
                    columns=["predictions", label_type],
                    index=labels_all.index,
                ),
                acu_dates=first_op_df,
                model_name=name,
                label_type=label_type,
                save_path=config.data.figures_path,
                right_censor=int(label_type.split("_")[1]) + 1,
            )

        # Create calibration plot
        calibration_plot(predictions, y_test, config.data.figures_path, label_type)

        # Create Net Benefit Curve
        net_benefit_plot(predictions, y_test, config.data.figures_path, label_type)


if __name__ == "__main__":
    main(random_state=config.seed)
