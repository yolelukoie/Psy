from data_loader import *
from analysis import *
from classification import *
from deep_model import *


def main():
    """load_all_data()
    run_statistical_tests('features.csv')
    plot_entropy_violin()
    plot_faa_violin()
    plot_heatmap_from_correlation()
    run_regression("features.csv")
    run_mixed_anova("features.csv")
    run_classification("features.csv", run_filter="01")  # run_filter="01" or "02" / too imbalanced for now
    run_classification("features.csv", run_filter="02")  # run_filter="01" or "02" / too imbalanced for now
    
    plot_run_comparison(SMOTE=False)

    permutation_test("features.csv", run_filter="01")
    permutation_test("features.csv", run_filter="02")

    feature_comparison("features.csv")

    run_deep_learning("features.csv", run_filter="01")
    run_deep_learning("features.csv", run_filter="02")

    plot_all_model_performance()"""

    # region ------ Topomaps ------

    plot_mean_band_topomaps("features.csv")
    run_all_cluster_tests("features.csv")
    plot_groupwise_correlation_heatmaps("features.csv")
    explain_with_shap("features.csv", run_filter="01")
    explain_with_shap("features.csv", run_filter="02")

    
    # endregion

if __name__ == "__main__":
    main()
    