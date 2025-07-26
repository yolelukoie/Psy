import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, spearmanr, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pingouin import mixed_anova
import mne
from mne.stats import permutation_cluster_test
import os
import re
import matplotlib.colorbar as cbar
from scipy.stats import ttest_ind



def run_statistical_tests(features_df):
    features_df = pd.read_csv("features.csv")
    features_df["run"] = features_df["run"].astype(str).str.zfill(2)
    clinical_vars = ["MDD", "BDI", "BDI_Anh", "BDI_Mel", "TAI"]
    eeg_vars = ["faa", "spectral_entropy"] + [col for col in features_df.columns if col.startswith(('alpha_', 'beta_', 'theta_', 'delta_'))]

    for run_id in ["01", "02"]:
        suffix = f"_run{run_id}"
        df = features_df[features_df["run"] == run_id].copy()

        results = []
        for eeg_var in eeg_vars:
            for clinical_var in clinical_vars:
                temp_df = df[[eeg_var, clinical_var]].dropna()
                # Spearman correlation
                rho, p_corr = spearmanr(temp_df[eeg_var], temp_df[clinical_var], nan_policy='omit')

                # Group comparison logic
                if clinical_var == "MDD":
                    df_filtered = temp_df[temp_df[clinical_var].isin([1, 2, 50])]
                    groups = {
                        "Current": df_filtered[df_filtered[clinical_var] == 1][eeg_var],
                        "Past": df_filtered[df_filtered[clinical_var] == 2][eeg_var],
                        "None": df_filtered[df_filtered[clinical_var] == 50][eeg_var]
                    }
                elif temp_df[clinical_var].nunique() > 2:
                    median_val = temp_df[clinical_var].median()
                    df_grouped = temp_df.copy()
                    df_grouped["bin"] = np.where(temp_df[clinical_var] > median_val, "High", "Low")
                    groups = {
                        "High": df_grouped[df_grouped["bin"] == "High"][eeg_var],
                        "Low": df_grouped[df_grouped["bin"] == "Low"][eeg_var]
                    }
                else:
                    continue

                if len(groups) == 2:
                    group_keys = list(groups.keys())
                    t_stat, p_ttest = ttest_ind(groups[group_keys[0]], groups[group_keys[1]], equal_var=False)
                else:
                    t_stat, p_ttest = np.nan, np.nan

                results.append({
                    "run": run_id,
                    "eeg_feature": eeg_var,
                    "clinical_score": clinical_var,
                    "spearman_r": rho,
                    "spearman_p": p_corr,
                    "ttest_t": t_stat,
                    "ttest_p": p_ttest
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"Results/eeg_clinical_stats{suffix}.csv", index=False)
        sig_df = results_df[(results_df["spearman_p"] < 0.05) | (results_df["ttest_p"] < 0.05)] # subset with significant results
        sig_df.to_csv(f"Results/eeg_clinical_stats_significant{suffix}.csv", index=False)
        print(f"🔎 Saved significant results: eeg_clinical_stats_significant{suffix}.csv")


def run_regression(features_csv="features.csv"):
    df_all = pd.read_csv(features_csv)
    df_all["run"] = df_all["run"].astype(str).str.zfill(2)

    all_results = []

    for run_id in ["01", "02"]:
        df = df_all[df_all["run"] == run_id].copy()

        # Filter to subjects with usable data
        df = df.dropna(subset=["BDI", "faa", "spectral_entropy", "sex", "age"])

        # Define group: 0 = CTL, 1 = DEP
        df["group_bdi"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
        df = df.dropna(subset=["group_bdi"])
        df["group_bdi"] = df["group_bdi"].astype(int)

        # Encode sex: M = 2, F = 1
        df["sex"] = df["sex"].map({1.0: 1, 2.0: 2})

        # Select EEG features to analyze
        exclude_cols = ["BDI", "group_bdi", "sex", "age", "participant_id", "run"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        run_results = []

        for feature in feature_cols:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            formula = f"{feature} ~ group_bdi + age + sex"

            try:
                model = smf.ols(formula, data=df).fit()
                coef = model.params.get("group_bdi", np.nan)
                pval = model.pvalues.get("group_bdi", np.nan)
                pval_sex = model.pvalues["sex"]
                pval_age = model.pvalues["age"]
                coef_sex = model.params["sex"]
                coef_age = model.params["age"]
                r2 = model.rsquared
            except Exception as e:
                print(f"Error with feature {feature}: {e}")
                continue

            run_results.append({
                "run": run_id,
                "feature": feature,
                "coef_group_bdi": model.params.get("group_bdi", np.nan),
                "pval_group_bdi": model.pvalues.get("group_bdi", np.nan),
                "coef_sex": model.params.get("sex", np.nan),
                "pval_sex": model.pvalues.get("sex", np.nan),
                "coef_age": model.params.get("age", np.nan),
                "pval_age": model.pvalues.get("age", np.nan),
                "r_squared": model.rsquared
            })

        df_run = pd.DataFrame(run_results)

        # Apply FDR correction
        if not df_run.empty:
            _, pvals_corr, _, _ = multipletests(df_run["pval_group_bdi"], method='fdr_bh')
            df_run["pval_fdr"] = pvals_corr
            df_run["significant_fdr"] = df_run["pval_fdr"] < 0.05

        all_results.append(df_run)

    df_all_results = pd.concat(all_results, ignore_index=True)
    df_all_results.to_csv("Results/regression_results.csv", index=False)
    sig_covariates = df_all_results[ 
        (df_all_results["pval_sex"] < 0.05) | (df_all_results["pval_age"] < 0.05) #negative coef - females score higher
    ]
    sig_covariates.to_csv("Results/regression_sex_age_significant.csv", index=False)
    print("Saved subset with significant age or sex effects to Results/regression_sex_age_significant.csv")
    print(f"Results saved to Results/regression_results.csv")


def run_mixed_anova(features_csv="features.csv"):
   
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)

    # Define BDI-based group
    df = df.dropna(subset=["BDI", "faa", "spectral_entropy"])
    df["group_bdi"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["group_bdi"])
    df["group_bdi"] = df["group_bdi"].astype(int)
    df["group_bdi"] = df["group_bdi"].map({0: "CTL", 1: "DEP"})

    # Convert run to pre-task and post-task
    df = df[df["run"].isin(["01", "02"])]
    df["run"] = df["run"].map({"01": "pre", "02": "post"})

    # Keep only participants with both runs
    counts = df["subject"].value_counts()
    eligible_subjects = counts[counts == 2].index
    df = df[df["subject"].isin(eligible_subjects)]

    print(f"Running mixed ANOVA on {len(eligible_subjects)} participants with both runs.")

    # FAA as the dependent variable
    df_faa = df[["subject", "run", "group_bdi", "faa"]].dropna()

    # Run the mixed ANOVA
    aov = mixed_anova(dv="faa", within="run", between="group_bdi", subject="subject", data=df_faa)
    aov.to_csv("Results/mixed_anova_faa.csv", index=False)
    print("✅ Mixed ANOVA results saved as mixed_anova_faa.csv")

    # Optional: visualize interaction
    sns.pointplot(data=df_faa, x="run", y="faa", hue="group_bdi", dodge=True, markers="o", capsize=0.1)
    plt.title("FAA: Run × Group Interaction")
    plt.savefig("Plots/faa_run_group_interaction.png", dpi=300)
    plt.clf()


# region ------ Plots ------

def plot_entropy_violin(features_df="features.csv"):
    df = pd.read_csv(features_df)
    df["run"] = df["run"].astype(str).str.zfill(2)

    # Define BDI-based group
    df["group_bdi"] = df["BDI"].apply(lambda x: "CTL" if x < 8 else ("DEP" if x > 13 else np.nan))
    df = df.dropna(subset=["group_bdi", "spectral_entropy"])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="run", y="spectral_entropy", hue="group_bdi", split=True, inner="quartile")
    plt.title("Spectral Entropy by Run and BDI Group")
    plt.xlabel("Run (01 = Pre-task, 02 = Post-task)")
    plt.ylabel("Spectral Entropy")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig("Plots/spectral_entropy_violin.png", dpi=300)

def plot_faa_violin(features_df="features.csv"):
    df = pd.read_csv(features_df)
    df["run"] = df["run"].astype(str).str.zfill(2)

    # Define BDI-based group
    df["group_bdi"] = df["BDI"].apply(lambda x: "CTL" if x < 8 else ("DEP" if x > 13 else np.nan))
    df = df.dropna(subset=["group_bdi", "faa"])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="run", y="faa", hue="group_bdi", split=True, inner="quartile")
    plt.title("FAA by Run and BDI Group")
    plt.xlabel("Run (01 = Pre-task, 02 = Post-task)")
    plt.ylabel("FAA (Frontal Alpha Asymmetry)")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig("Plots/faa_violin.png", dpi=300)


def get_region(feature_name):
    feature_name = feature_name.lower()
    if "spectral_entropy" in feature_name:
        return "Global"
    if "_ch_" not in feature_name:
        return "Unknown"

    ch = feature_name.split("_ch_")[-1].upper()

    if ch.startswith(("FP", "AF", "F")):
        return "Frontal"
    elif ch.startswith(("FC", "C", "CZ")):
        return "Central"
    elif ch.startswith(("CP", "P", "PO")):
        return "Parietal"
    elif ch.startswith(("T", "TP")):
        return "Temporal"
    elif ch.startswith("O"):
        return "Occipital"
    else:
        return "Unknown"

def get_band(feature_name):
    if feature_name.startswith("delta"):
        return "Delta"
    elif feature_name.startswith("theta"):
        return "Theta"
    elif feature_name.startswith("alpha"):
        return "Alpha"
    elif feature_name.startswith("beta"):
        return "Beta"
    elif "spectral_entropy" in feature_name:
        return "Entropy"
    else:
        return "Unknown"

def plot_heatmap_from_correlation():
    df1 = pd.read_csv("Results/eeg_clinical_stats_significant_run01.csv")
    df2 = pd.read_csv("Results/eeg_clinical_stats_significant_run02.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    # Add band and region
    df["band"] = df["eeg_feature"].apply(get_band)
    df["region"] = df["eeg_feature"].apply(get_region)
    df["group"] = df["band"] + " – " + df["region"]

    # Compute mean correlation per (band–region) group and clinical score
    grouped = df.groupby(["group", "clinical_score"])["spearman_r"].mean().unstack()
    grouped = grouped.fillna(0)

    sns.set_theme(style="white", font_scale=0.9)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        grouped,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean Spearman r"}
    )

    plt.title("EEG Region-Frequency Correlation with Clinical Scores", pad=20)
    plt.tight_layout()
    plt.savefig("Plots/eeg_clinical_heatmap.png", dpi=300)
    print("✅ Grouped heatmap saved: Plots/eeg_clinical_heatmap.png")


def plot_all_model_performance():
    models = {
        "RF": "classification_results",
        "SMOTE": "classification_results_smote",
        "CNN": "deep_learning_results"
    }
    runs = ["run01", "run02"]

    all_dfs = []
    for model_name, prefix in models.items():
        for run in runs:
            path = f"Results/{prefix}_{run}.csv"
            df = pd.read_csv(path)
            df["run"] = run
            df["model"] = model_name
            all_dfs.append(df)

    df_combined = pd.concat(all_dfs)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_combined, x="metric", y="mean", hue="model", ci=None)
    plt.title("Model Comparison Across Metrics (per run)")
    plt.ylabel("Score")
    plt.ylim(0.4, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("Plots/model_comparison_all.png", dpi=300)
    plt.clf()
    print("✅ Saved: Plots/model_comparison_all.png")

def plot_top_features_as_topomap(feature_importance_csv, band_prefix="alpha", ax=None):
    df = pd.read_csv(feature_importance_csv)
    top_band = df[df["feature"].str.startswith(f"{band_prefix}_ch_")].copy()
    top_band["channel"] = top_band["feature"].str.replace(f"{band_prefix}_ch_", "", regex=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names

    data = np.zeros(len(ch_names))
    ch_pos_dict = montage.get_positions()["ch_pos"]
    pos = []

    for ch in ch_names:
        pos.append(ch_pos_dict.get(ch, [np.nan, np.nan])[:2])
        if ch in top_band["channel"].values:
            data[ch_names.index(ch)] = top_band.loc[top_band["channel"] == ch, "importance"].values[0]

    data = np.array(data)
    pos = np.array(pos)

    if np.all(data == 0):
        print(f"⚠️ No importance data for band {band_prefix}")
        return

    fig, im = mne.viz.plot_topomap(data, pos, axes=ax, contours=0)
    for name, xy in zip(ch_names, pos):
        ax.text(xy[0], xy[1], name, fontsize=7, ha='center', va='center')
    ax.set_title(f"{band_prefix.capitalize()} Band", fontsize=12)

def plot_all_band_topomaps(feature_importance_csv, bands=["alpha", "beta", "theta"]):
    # Extract run and band from filename
    base = os.path.basename(feature_importance_csv)
    run_match = re.search(r'run\d{2}', base)
    run_suffix = run_match.group(0) if run_match else "unknown"

    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 5))

    for i, band in enumerate(bands):
        try:
            plot_top_features_as_topomap(feature_importance_csv, band_prefix=band, ax=axes[i])
        except Exception as e:
            print(f"⚠️ Could not plot {band} band: {e}")
            axes[i].set_visible(False)

    plt.suptitle(f"EEG Feature Importance Topomaps ({run_suffix})", fontsize=14)
    plt.tight_layout()

    output_path = f"Plots/topomap_combined_{run_suffix}_{band}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved combined topomap: {output_path}")
# endregion


# region ------ Mean powers ------
def plot_mean_band_topomaps(features_csv="features.csv"):
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df["group_bdi"] = df["BDI"].apply(lambda x: "CTL" if x < 8 else ("DEP" if x > 13 else np.nan))
    df = df.dropna(subset=["group_bdi"])

    bands = ["delta", "theta", "alpha", "beta"]
    runs = ["01", "02"]
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names
    pos = np.array([montage.get_positions()["ch_pos"].get(ch, [np.nan, np.nan])[:2] for ch in ch_names])

    for run in runs:
        df_run = df[df["run"] == run]

        fig, axes = plt.subplots(3, len(bands), figsize=(4.5 * len(bands), 12), constrained_layout=True)

        for col_i, band in enumerate(bands):
            ch_cols = [col for col in df_run.columns if col.startswith(f"{band}_ch_")]
            if not ch_cols:
                print(f"⚠️ No data for band {band} in run {run}")
                continue

            # --- CTL ---
            ctl_df = df_run[df_run["group_bdi"] == "CTL"]
            data_ctl = np.zeros(len(ch_names))
            for idx, ch in enumerate(ch_names):
                col = f"{band}_ch_{ch}"
                if col in ch_cols:
                    data_ctl[idx] = ctl_df[col].mean()
            mne.viz.plot_topomap(data_ctl, pos, axes=axes[0, col_i], contours=0, show=False)
            axes[0, col_i].set_title(f"{band.capitalize()} – CTL")

            # --- DEP ---
            dep_df = df_run[df_run["group_bdi"] == "DEP"]
            data_dep = np.zeros(len(ch_names))
            for idx, ch in enumerate(ch_names):
                col = f"{band}_ch_{ch}"
                if col in ch_cols:
                    data_dep[idx] = dep_df[col].mean()
            mne.viz.plot_topomap(data_dep, pos, axes=axes[1, col_i], contours=0, show=False)
            axes[1, col_i].set_title(f"{band.capitalize()} – DEP")

            # --- Difference (DEP - CTL) ---
            diff = data_dep - data_ctl
            mne.viz.plot_topomap(diff, pos, axes=axes[2, col_i], contours=0, show=False)
            axes[2, col_i].set_title(f"{band.capitalize()} – DEP minus CTL")

        axes[0, 0].set_ylabel("CTL", fontsize=12)
        axes[1, 0].set_ylabel("DEP", fontsize=12)
        axes[2, 0].set_ylabel("DEP – CTL", fontsize=12)

        plt.suptitle(f"Mean Band Power Topomaps by Group (Run {run})", fontsize=16)
        output_path = f"Plots/mean_band_power_all_topomaps_run{run}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved mean power scalp maps: {output_path}")

from scipy.stats import ttest_ind

def run_all_cluster_tests(features_csv="features.csv"):
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df["group_bdi"] = df["BDI"].apply(lambda x: 0 if x < 8 else (1 if x > 13 else np.nan))
    df = df.dropna(subset=["group_bdi"])

    bands = ["delta", "theta", "alpha", "beta"]
    runs = ["01", "02"]

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names
    pos = np.array([montage.get_positions()["ch_pos"].get(ch, [np.nan, np.nan])[:2] for ch in ch_names])

    # Custom t-stat function for 2-sided comparison
    def stat_fun(X, Y):
        t_vals, _ = ttest_ind(X, Y, axis=0, equal_var=False)
        return t_vals

    for run in runs:
        df_run = df[df["run"] == run]
        print(f"\n🔬 Run {run}: cluster testing...")

        for band in bands:
            ch_cols = [col for col in df_run.columns if col.startswith(f"{band}_ch_")]
            if not ch_cols:
                print(f"⚠️ No channels found for {band} in run {run}")
                continue

            X = df_run[ch_cols].values
            labels = df_run["group_bdi"].values

            if len(np.unique(labels)) < 2:
                print(f"❌ Not enough groups for {band} Run {run}")
                continue

            X_ctl = X[labels == 0]
            X_dep = X[labels == 1]

            if X_ctl.shape[0] < 5 or X_dep.shape[0] < 5:
                print(f"⚠️ Not enough subjects in one of the groups for {band} Run {run}")
                continue

            # Run cluster-based permutation t-test
            T_obs, clusters, p_values, _ = permutation_cluster_test(
                [X_ctl, X_dep],
                n_permutations=1000,
                stat_fun=stat_fun,
                tail=0,  # 0 = two-sided
                out_type='mask',
                verbose=False
            )

            # Create significance mask
            sig_mask = np.zeros(X_ctl.shape[1], dtype=bool)
            for i, p_val in enumerate(p_values):
                if p_val < 0.05:
                    sig_mask |= clusters[i]

            if not np.any(sig_mask):
                print(f"🚫 No significant clusters for {band} Run {run}")
                continue

            # Match features to standard montage channels
            col_ch_names = [col.split("_ch_")[1] for col in ch_cols]
            data = np.zeros(len(ch_names))
            for i, ch in enumerate(ch_names):
                if ch in col_ch_names:
                    idx = col_ch_names.index(ch)
                    data[i] = sig_mask[idx].astype(float)

            fig, ax = plt.subplots()
            mne.viz.plot_topomap(data, pos, axes=ax, contours=0, show=False)
            ax.set_title(f"Sig. Cluster Mask: {band.capitalize()} Run {run}")
            fig.savefig(f"Plots/cluster_mask_{band}_run{run}.png", dpi=300)
            plt.close(fig)
            print(f"✅ Saved: Plots/cluster_mask_{band}_run{run}.png")


def plot_groupwise_correlation_heatmaps(features_csv="features.csv"):
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df = df[df["run"].isin(["01", "02"])]  # combine both runs
    df["group_bdi"] = df["BDI"].apply(lambda x: "CTL" if x < 8 else ("DEP" if x > 13 else np.nan))
    df = df.dropna(subset=["group_bdi"])

    eeg_vars = [col for col in df.columns if col.startswith(("alpha_", "beta_", "theta_", "delta_"))]
    clinical_scores = ["BDI", "BDI_Anh", "BDI_Mel", "TAI"]

    def get_group_corr(df_group):
        corr_matrix = []
        for eeg in eeg_vars:
            row = []
            for clinical in clinical_scores:
                sub = df_group[[eeg, clinical]].dropna()
                if len(sub) >= 5:
                    rho, _ = spearmanr(sub[eeg], sub[clinical])
                else:
                    rho = np.nan
                row.append(rho)
            corr_matrix.append(row)
        return pd.DataFrame(corr_matrix, index=eeg_vars, columns=clinical_scores)

    df_ctl = df[df["group_bdi"] == "CTL"]
    df_dep = df[df["group_bdi"] == "DEP"]

    ctl_corr = get_group_corr(df_ctl)
    dep_corr = get_group_corr(df_dep)

    # Plot CTL
    plt.figure(figsize=(12, 10))
    sns.heatmap(ctl_corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Spearman Correlation: CTL Group")
    plt.tight_layout()
    plt.savefig("Plots/group_corr_heatmap_CTL.png", dpi=300)
    plt.close()

    # Plot DEP
    plt.figure(figsize=(12, 10))
    sns.heatmap(dep_corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Spearman Correlation: DEP Group")
    plt.tight_layout()
    plt.savefig("Plots/group_corr_heatmap_DEP.png", dpi=300)
    plt.close()

    print("✅ Saved group-specific correlation heatmaps:")
    print("   - Plots/group_corr_heatmap_CTL.png")
    print("   - Plots/group_corr_heatmap_DEP.png")

