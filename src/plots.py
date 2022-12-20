import os
from typing import Dict, List
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.calibration import calibration_curve
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from sklearn.linear_model import LogisticRegression
from lifelines.statistics import logrank_test

from src.metrics import net_benefit_curve, treat_all_curve, treat_none_curve


plt.rcParams["axes.facecolor"] = "w"
plt.rcParams["figure.facecolor"] = "w"
plt.style.use("seaborn-colorblind")

import cycler

jama_style_colors = [
    "#374E55FF",
    "#DF8F44FF",
    "#00A1D5FF",
    "#B24745FF",
    "#79AF97FF",
    "#6A6599FF",
    "#80796BFF",
]

plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", jama_style_colors)


def calibration_plot(
    preds: Dict, y_true: np.ndarray, save_path: str, label_type: str
) -> None:
    """Creates calibration plot for a list of models and their names
    Args:
        preds (Dict): dictionary of predictions with model name as key and risk predictions of the test set as value
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
        label_type (str): type of the label
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=1, figsize=(8, 6))

    # Plot perfectly calibrated
    ax = axes
    for model, y_pred in preds.items():
        x, y = calibration_curve(y_true, y_pred, n_bins=20)

        # Plot model's calibration curve
        ax.plot(y, x, marker=".", label=model)

    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideally Calibrated")
    ax.set_title(f"Calibration Curve for {label_type} models")
    ax.legend()
    ax.set_xlabel("Average Predicted Probability in each bin")
    ax.set_ylabel("Ratio of positives")
    ax.grid()
    plt.savefig(
        os.path.join(save_path, f"calibrations_{label_type}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def net_benefit_plot(
    preds: Dict, y_true: np.ndarray, save_path: str, label_type: str
) -> None:
    """Creates net benefit plot for a list of models and their names
    Args:
        preds (Dict): dictionary of predictions with model name as key and risk predictions of the test set as value
        y_true (np.ndarray): numpy array with the true labels
        save_path (str): path where the figure will be stored
        label_type (str): type of the label
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))

    ax = axes
    for model, y_pred in preds.items():
        net_benefit_curve(ax, y_true, y_pred, title=model)
    treat_all_curve(ax, y_true, lowest_net_benefit=-0.2, title="Treat All")
    treat_none_curve(ax, title="Treat None")

    ax.set_title(f"Net Benefit for {label_type} models")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=-0.1)
    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Net Benefit Score")
    ax.grid()
    plt.savefig(
        os.path.join(save_path, f"net_benefit_{label_type}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_group_sensitivity(
    predictions: np.ndarray,
    feature_matrix: pd.DataFrame,
    names_dict: Dict,
    group_name: str,
    save_path: str,
    rotation: int = 0,
    order: bool = False,
) -> None:
    """Plot senstivity analysis for group uncertainty. Creates box plots with 0.25, 0.5, and 0.75 quantile, as well as outliers, according to groups
    NOTE: this works only for group > 2. If you wish to compare a single one-hot encoded group, please refer to `plot_group_uncertainty_binary`.
    Args:
        predictions (np.ndarray): risk predictions of a model
        feature_matrix (pd.DataFrame): dataframe containing the feautures (unnormalized) with the corresponding columns
        names_dict (Dict): dictionary containing the feature matrix column names as keys and the formatted clean names as values
        group_name (str): name of the group that is being inspected
        save_path (str): path where the figure will be stored
        rotation (int, 0): roation of the names labels
        order (bool, False): keeps order of the boxes
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    data_df = feature_matrix[names_dict.keys()]
    data_df["pred_percentile"] = pd.qcut(predictions, 100).codes + 1
    data_df.rename(
        columns={x: f"{y} (n={data_df[x].sum()})" for x, y in names_dict.items()},
        inplace=True,
    )

    # reverse one-hot encoding
    data_df[group_name] = (data_df.iloc[:, : len(names_dict)] == 1).idxmax(1)

    ax.plot(
        [0, 100],
        [0, 1],
        color="grey",
        linewidth=2,
        ls="--",
        alpha=0.8,
        label=f"No Impact of {group_name}",
    )
    leg1 = ax.legend(loc="lower right")
    sns.ecdfplot(
        data=data_df,
        x="pred_percentile",
        hue=group_name,
        hue_order=data_df.columns[: len(names_dict)] if order else None,
        ax=ax,
        legend=True,
        complementary=False,
    )
    plt.xlabel("Percentile of Algorithm Risk Score")
    plt.ylabel("Cumulative Proportion of Group")
    ax.add_artist(leg1)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=None))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=None))
    plt.xlim([0, 100])

    plt.title(f"Cumulative Risk by {group_name}")
    plt.grid()
    plt.xticks(rotation=rotation)
    plt.savefig(
        os.path.join(
            save_path, f"sensitivity_{'_'.join(group_name.lower().split())}.pdf"
        ),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_group_uncertainty_binary(
    predictions: np.ndarray,
    feature_matrix: pd.DataFrame,
    col_group_name: str,
    group_name: str,
    binary_dict: Dict,
    save_path: str,
) -> None:
    """Plot senstivity analysis for group uncertainty. Creates box plots with 0.25, 0.5, and 0.75 quantile, as well as outliers, according to groups
    NOTE: this works only for group == 2. If you wish to compare multiple one-hot encoded group, please refer to `plot_group_uncertainty`.
    Args:
        predictions (np.ndarray):  risk predictions of a model
        feature_matrix (pd.DataFrame): dataframe containing the feautures (unnormalized) with the corresponding columns
        col_group_name (str): column name of the group to be inspected
        group_name (str): name of the group that is being inspected
        binary_dict (Dict): Dictionary mapping `0` to the proper label and `1` to the proper label for one hot encoding
        save_path (str): path where the figure will be stored
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    data_df = feature_matrix[[col_group_name]]
    yes = int(data_df[col_group_name].sum())
    no = int(len(data_df[col_group_name]) - data_df[col_group_name].sum())
    data_df[col_group_name] = data_df[col_group_name].map(
        {1: f"{binary_dict[1]} (n={yes})", 0: f"{binary_dict[0]} (n={no})"}
    )
    data_df["pred_percentile"] = pd.qcut(predictions, 100).codes + 1
    data_df.rename(columns={col_group_name: group_name}, inplace=True)

    ax.plot(
        [0, 100],
        [0, 1],
        color="grey",
        linewidth=2,
        ls="--",
        alpha=0.8,
        label=f"No Impact of {group_name}",
    )
    leg1 = ax.legend(loc="lower right")
    sns.ecdfplot(
        data=data_df,
        x="pred_percentile",
        hue=group_name,
        ax=ax,
        legend=True,
        complementary=False,
    )
    plt.xlabel("Percentile of Algorithm Risk Score")
    plt.ylabel("Cumulative Proportion of Group")
    ax.add_artist(leg1)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=None))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=None))
    plt.xlim([0, 100])

    plt.title(f"Cumulative Risk by {group_name}")
    plt.grid()
    plt.xticks()
    plt.savefig(
        os.path.join(
            save_path, f"sensitivity_{'_'.join(group_name.lower().split())}.pdf"
        ),
        dpi=300.0,
        bbox_inches="tight",
    )


def kaplan_meyer_plot(
    predictions: np.ndarray,
    acu_dates: pd.DataFrame,
    model_name: str,
    label_type: str,
    save_path: str,
    right_censor: int = 181,
) -> None:
    """Plot Kaplan-Meyer curves for single models
    Args:
        predictions (np.ndarray): array with the risk predictions
        acu_dates (pd.DataFrame): dataframe containing the first date of acute care use
        model_name (str): name of the model
        label_type (str): type of label used
        save_path (str): path where the figure will be stored
        right_censor (int, 181): right censor date
    Returns:
        None
    """
    acu_dates["CHE_TO_HOSP"] = (
        acu_dates["CONTACT_DATE"] - acu_dates["CHE_TX_DATE"]
    ).dt.days
    acu_dates = acu_dates.sort_values(by=["PAT_DEID", "CONTACT_DATE"]).drop_duplicates(
        subset="PAT_DEID", keep="first"
    )
    acu_dates = acu_dates[["PAT_DEID", "CHE_TO_HOSP"]]
    acu_dates.columns = ["PAT_DEID", "CHE_TO_EVENT"]

    # join these data sets and bin by probability tertile (e.g. 0.9 = 90% prob of having event, would be high-risk tertile)
    predictions = pd.merge(predictions, acu_dates, how="left", on="PAT_DEID")
    predictions["tertile"] = pd.qcut(predictions["predictions"], q=3, labels=False)

    # right-censor at 180d
    predictions["CHE_TO_EVENT"] = (
        predictions["CHE_TO_EVENT"]
        .replace(np.nan, right_censor)
        .clip(upper=right_censor)
    ) + 0.5

    # collect by tertile
    predictions_1 = predictions.query("tertile==0")
    predictions_2 = predictions.query("tertile==1")
    predictions_3 = predictions.query("tertile==2")

    # declare and fit KMF
    kmf_1 = KaplanMeierFitter(label="Low Risk")
    kmf_2 = KaplanMeierFitter(label="Intermediate Risk")
    kmf_3 = KaplanMeierFitter(label="High Risk")
    kmf_1.fit(
        durations=predictions_1["CHE_TO_EVENT"],
        event_observed=predictions_1[label_type],
    )
    kmf_2.fit(
        durations=predictions_2["CHE_TO_EVENT"],
        event_observed=predictions_2[label_type],
    )
    kmf_3.fit(
        durations=predictions_3["CHE_TO_EVENT"],
        event_observed=predictions_3[label_type],
    )

    # Logrank Test:
    lrt_1 = logrank_test(
        predictions_1["CHE_TO_EVENT"],
        predictions_2["CHE_TO_EVENT"],
        event_observed_A=predictions_1[label_type],
        event_observed_B=predictions_2[label_type],
    )

    lrt_2 = logrank_test(
        predictions_2["CHE_TO_EVENT"],
        predictions_3["CHE_TO_EVENT"],
        event_observed_A=predictions_2[label_type],
        event_observed_B=predictions_3[label_type],
    )

    lrt_3 = logrank_test(
        predictions_1["CHE_TO_EVENT"],
        predictions_3["CHE_TO_EVENT"],
        event_observed_A=predictions_1[label_type],
        event_observed_B=predictions_3[label_type],
    )

    p_value = (
        "p < 0.001"
        if lrt_1.p_value < 0.001 and lrt_2.p_value < 0.001 and lrt_3.p_value < 0.001
        else f"p = {max(lrt_1.p_value, lrt_2.p_value,lrt_3.p_value):.3f}"
    )

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf_1.plot(color=jama_style_colors[4])
    kmf_2.plot(color=jama_style_colors[1])
    kmf_3.plot(color=jama_style_colors[3])
    plt.xlim([0, right_censor - 1])
    plt.ylim([0, 1.0])
    plt.legend(loc=3)
    plt.ylabel("Percent without Acute Care Use")
    plt.xlabel("Days Since First Chemotherapy")
    plt.title(
        f"Risk-Stratified Kaplan-Meier Surival Estimates for Acute Care Use\n{model_name}\nHighest LogRank test: {p_value}"
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(np.arange(0, right_censor, (right_censor - 1) // 6))
    plt.grid()
    add_at_risk_counts(kmf_1, kmf_2, kmf_3, rows_to_show=["Events"])
    plt.savefig(
        os.path.join(save_path, f"KM_{model_name}_{label_type}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_word_importance(
    model: LogisticRegression,
    data_columns: List,
    label_type: str,
    save_path: str,
    n: int = 15,
) -> None:
    """Plot the `n` highest and lowest words based on their odds ratio
    Args:
        model (LogisticRegression): sklearn logistic LASSO
        data_columns (List): List of data columns
        save_path (str): path where the figure will be stored
        label_type (str): type of label used
        n (integer, 15): n highest and lowest coefficients
    Returns:
        None
    """
    odds_ratios = model.coef_[0]
    index_order = np.argsort(odds_ratios)
    odds_ratios = odds_ratios[index_order].tolist()
    data_columns = data_columns[index_order].tolist()

    odds_ratios = np.array([*odds_ratios[:n], *odds_ratios[-n:]])
    data_columns = np.array([*data_columns[:n], *data_columns[-n:]])
    c = ["#DF8F44FF"] * n + ["#374E55FF"] * n
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.barh(np.arange(len(odds_ratios)), odds_ratios, color=c)
    ax.set_xlabel("Coefficient Magnitude")
    ax.set_yticks(
        np.arange(len(data_columns)),
        labels=["_".join(word.split("_")[1:]).capitalize() for word in data_columns],
    )
    plt.axvline(x=0)
    ax.grid()
    plt.title("Word importance for using Language LASSO:\n OP-35 Event within 180 days")
    plt.savefig(
        os.path.join(save_path, f"word_importance_{label_type}.pdf"),
        dpi=300.0,
        bbox_inches="tight",
    )


def plot_prediction_comparison(
    predictions: Dict,
    feature_matrix: pd.DataFrame,
    names_dict: Dict,
    group_name: str,
    save_path: str,
    label_type: str,
) -> None:
    """Creates a scatter plot of model 1 and model 2 predictions, and colours the scatters according to the group
    Args:
        predictions (Dict): name and predictions of the first and second model {name (str): predictions (np.ndarray)}
        feature_matrix (pd.DataFrame): data frame, with columns containing all the input features
        names_dict (Dict): dictionary containing the feature matrix column names as keys and the formatted clean names as values
        group_name (str): name of the groups inspected
        save_path (str): path where the figure will be stored
        label_type (str): label_type
    Returns:
        None"""
    model_1_name = list(predictions.keys())[0]
    model_2_name = list(predictions.keys())[1]
    model_1_preds = list(predictions.values())[0]
    model_2_preds = list(predictions.values())[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    data_df = feature_matrix[names_dict.keys()]
    data_df[model_1_name] = model_1_preds
    data_df[model_2_name] = model_2_preds
    data_df.rename(
        columns={x: f"{y} (n={data_df[x].sum()})" for x, y in names_dict.items()},
        inplace=True,
    )

    # reverse one-hot encoding
    data_df[group_name] = (data_df.iloc[:, : len(names_dict)] == 1).idxmax(1)

    ax.plot(
        [0, 1],
        [0, 1],
        color="grey",
        linewidth=2,
        ls="--",
        alpha=0.8,
        label=f"Same Risk Predictions",
    )
    sns.scatterplot(
        data=data_df,
        x=model_1_name,
        y=model_2_name,
        hue=group_name,
        ax=ax,
        legend=True,
    )
    ax.grid()
    ax.set_title(
        f"ACU risk predictions of {model_1_name} vs. {model_2_name}\nsubdivided by {group_name}"
    )
    ax.set_xlabel(f"Predictions {model_1_name}")
    ax.set_ylabel(f"Predictions {model_2_name}")
    plt.legend()
    plt.savefig(
        os.path.join(
            save_path,
            f"{'_'.join(model_1_name.lower().split())}_vs_{'_'.join(model_2_name.lower().split())}_{label_type}.pdf",
        ),
        dpi=300.0,
        bbox_inches="tight",
    )
