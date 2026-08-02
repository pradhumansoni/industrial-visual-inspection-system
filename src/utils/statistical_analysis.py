"""
=========================================================
Statistical Analysis Utilities
=========================================================

Reusable statistical functions for the EDA notebooks.

Author : VisionInspect AI
"""

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    shapiro,
    levene,
    f_oneway,
    kruskal,
)

# Plot Boxplots 

def plot_boxplot(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
    figsize=(10, 6),
):
    """
    Plot a boxplot of a numerical feature grouped by category.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    feature : str
        Numerical feature to visualize.

    group : str, default="class"
        Grouping column.

    figsize : tuple
        Figure size.
    """

    plt.figure(figsize=figsize)

    data.boxplot(
        column=feature,
        by=group,
        grid=False,
    )

    plt.title(
        f"{feature.replace('_', ' ').title()} by {group.title()}"
    )

    plt.suptitle("")

    plt.xlabel(group.title())

    plt.ylabel(
        feature.replace("_", " ").title()
    )

    plt.show()

# Run Shapiro Wilk Test To Check Normality in each class

def shapiro_test(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
):
    """
    Perform Shapiro-Wilk normality test
    independently for every group.
    """

    results = []

    for cls, values in data.groupby(group):

        statistic, p_value = shapiro(
            values[feature]
        )

        results.append(
            {
                "Class": cls,
                "Statistic": statistic,
                "p-value": p_value,
                "Normal": p_value > 0.05,
            }
        )

    return pd.DataFrame(results)

# Run Levene Test To check Homegenity of Variance between the classes

def levene_test(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
):
    """
    Perform Levene's test for homogeneity of variances.
    """

    groups = [
        values[feature].values
        for _, values in data.groupby(group)
    ]

    statistic, p_value = levene(*groups)

    return statistic, p_value

# Run One-Way ANOVA To Check if all the classes are almost similar with respect to the Statistic 

def anova_test(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
):
    """
    Perform one-way ANOVA.
    """

    groups = [
        values[feature].values
        for _, values in data.groupby(group)
    ]

    statistic, p_value = f_oneway(*groups)

    return statistic, p_value

# Run Kruskal Wallis Test (If One-Way ANOVA Assumptions Fails)

def kruskal_test(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
):
    """
    Perform Kruskal-Wallis H test.
    """

    groups = [
        values[feature].values
        for _, values in data.groupby(group)
    ]

    statistic, p_value = kruskal(*groups)

    return statistic, p_value

# Entire Pipeline 

def compare_feature_between_classes(data, feature, group="class"):
    """
    Complete statistical comparison pipeline.
    """

    print("=" * 70)
    print(feature.upper())
    print("=" * 70)

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    plot_feature_distribution(data, feature, group)

    # -----------------------------------------------------
    # Shapiro
    # -----------------------------------------------------

    print("\nShapiro-Wilk Test")
    print("-" * 70)

    shapiro_results = shapiro_test(data, feature, group)

    display(shapiro_results)

    normal = shapiro_results["Normal"].all()

    # -----------------------------------------------------
    # Levene
    # -----------------------------------------------------

    statistic, p_value = levene_test(data, feature, group)

    print("\nLevene's Test")
    print("-" * 70)

    print(f"Statistic : {statistic:.4f}")
    print(f"P-value   : {p_value:.4f}")

    equal_variance = p_value > 0.05

    # -----------------------------------------------------
    # Choose Test
    # -----------------------------------------------------

    if normal and equal_variance:

        print("\nUsing One-Way ANOVA")

        statistic, p_value = anova_test(
            data,
            feature,
            group
        )

        test_name = "ANOVA"

    else:

        print("\nUsing Kruskal-Wallis Test")

        statistic, p_value = kruskal_test(
            data,
            feature,
            group
        )

        test_name = "Kruskal-Wallis"

    print("-" * 70)
    print(f"{test_name} Statistic : {statistic:.4f}")
    print(f"P-value             : {p_value:.6f}")

    if p_value < 0.05:

        print("\nConclusion:")
        print("Statistically significant difference detected.")

    else:

        print("\nConclusion:")
        print("No statistically significant difference detected.")

    return {
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value
    }


def compare_groups(
    data: pd.DataFrame,
    feature: str,
    group: str = "class",
    alpha: float = 0.05,
    visualize: bool = True,
):
    """
    Compare a numerical feature across multiple groups.

    Workflow
    --------
    1. Boxplot
    2. Shapiro-Wilk Test
    3. Levene's Test
    4. Automatically select:
        - One-Way ANOVA
        - Kruskal-Wallis
    """

    # -----------------------------------------------------
    # Boxplot
    # -----------------------------------------------------

    if visualize:
        plot_boxplot(
            data=data,
            feature=feature,
            group=group,
        )

    # -----------------------------------------------------
    # Shapiro
    # -----------------------------------------------------

    shapiro_results = shapiro_test(
        data=data,
        feature=feature,
        group=group,
    )

    normality_passed = shapiro_results["Normal"].all()

    failed_classes = shapiro_results.loc[
        ~shapiro_results["Normal"],
        "Class"
    ].tolist()

    # -----------------------------------------------------
    # Levene
    # -----------------------------------------------------

    levene_statistic, levene_pvalue = levene_test(
        data=data,
        feature=feature,
        group=group,
    )

    equal_variance = levene_pvalue > alpha

    # -----------------------------------------------------
    # Statistical Test
    # -----------------------------------------------------

    if normality_passed and equal_variance:

        selected_test = "One-Way ANOVA"

        test_statistic, test_pvalue = anova_test(
            data=data,
            feature=feature,
            group=group,
        )

    else:

        selected_test = "Kruskal-Wallis"

        test_statistic, test_pvalue = kruskal_test(
            data=data,
            feature=feature,
            group=group,
        )

    return {

        "feature": feature,

        "alpha": alpha,

        "num_groups": data[group].nunique(),

        "num_samples": len(data),

        "shapiro": shapiro_results,

        "normality_passed": normality_passed,

        "failed_classes": failed_classes,

        "levene_statistic": levene_statistic,

        "levene_pvalue": levene_pvalue,

        "equal_variance": equal_variance,

        "selected_test": selected_test,

        "test_statistic": test_statistic,

        "test_pvalue": test_pvalue,

    }


def print_statistical_report(results: dict):

    print("=" * 70)
    print("STATISTICAL COMPARISON REPORT")
    print("=" * 70)

    print("\nFeature Information")
    print("-" * 70)

    print(f"Feature               : {results['feature']}")
    print(f"Number of Groups      : {results['num_groups']}")
    print(f"Total Samples         : {results['num_samples']}")
    print(f"Significance Level    : α = {results['alpha']}")

    # ---------------------------------------------------------
    # Shapiro
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 1 : NORMALITY ASSUMPTION")
    print("=" * 70)

    print("\nTest")
    print("-" * 70)
    print("Shapiro-Wilk Test")

    print("\nPurpose")
    print("-" * 70)
    print("Checks whether each group's observations")
    print("are approximately normally distributed.")

    print("\nDecision Rule")
    print("-" * 70)
    print("Reject H₀ if p-value < α")

    display(results["shapiro"])

    passed = results["shapiro"]["Normal"].sum()
    failed = len(results["shapiro"]) - passed

    print("\nSummary")
    print("-" * 70)

    print(f"Groups Passed         : {passed}")
    print(f"Groups Failed         : {failed}")

    if results["failed_classes"]:

        print("\nFailed Classes")

        for cls in results["failed_classes"]:
            print(f"• {cls}")

    print("\nConclusion")
    print("-" * 70)

    if results["normality_passed"]:
        print("✓ Normality assumption satisfied.")
    else:
        print("✗ Normality assumption NOT satisfied.")

    # ---------------------------------------------------------
    # Levene
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2 : HOMOGENEITY OF VARIANCE")
    print("=" * 70)

    print("\nTest")
    print("-" * 70)
    print("Levene's Test")

    print("\nPurpose")
    print("-" * 70)
    print("Checks whether all groups have equal variance.")

    print("\nDecision Rule")
    print("-" * 70)
    print("Reject H₀ if p-value < α")

    print(f"\nStatistic            : {results['levene_statistic']:.4f}")
    print(f"P-value              : {results['levene_pvalue']:.6f}")

    print("\nConclusion")
    print("-" * 70)

    if results["equal_variance"]:
        print("✓ Equal variance assumption satisfied.")
    else:
        print("✗ Equal variance assumption NOT satisfied.")

    # ---------------------------------------------------------
    # Test Selection
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3 : TEST SELECTION")
    print("=" * 70)

    if results["selected_test"] == "One-Way ANOVA":

        print("\nPreferred Test")
        print("-" * 70)
        print("One-Way ANOVA")

        print("\nStatus")
        print("-" * 70)
        print("SELECTED ✓")

        print("\nReason")
        print("-" * 70)
        print("All ANOVA assumptions are satisfied.")

    else:

        print("\nPreferred Test")
        print("-" * 70)
        print("One-Way ANOVA")

        print("\nStatus")
        print("-" * 70)
        print("SKIPPED")

        print("\nReason")
        print("-" * 70)

        if not results["normality_passed"]:
            print("• Normality assumption violated.")

        if not results["equal_variance"]:
            print("• Equal variance assumption violated.")

        print("\nSelected Test")
        print("-" * 70)
        print("Kruskal-Wallis")

    # ---------------------------------------------------------
    # Final Test
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4 : HYPOTHESIS TEST")
    print("=" * 70)

    print(f"\nTest Performed       : {results['selected_test']}")

    print("\nNull Hypothesis")
    print("-" * 70)
    print("All groups have identical distributions.")

    print("\nAlternative Hypothesis")
    print("-" * 70)
    print("At least one group differs.")

    print(f"\nStatistic            : {results['test_statistic']:.4f}")
    print(f"P-value              : {results['test_pvalue']:.6f}")

    print("\nDecision")
    print("-" * 70)

    if results["test_pvalue"] < results["alpha"]:
        print("Reject H₀")
        print("Statistically significant difference detected.")
    else:
        print("Fail to Reject H₀")
        print("No statistically significant difference detected.")

    print("=" * 70)

