from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# region ------ classification functions ------
def run_classification(features_csv="features.csv", run_filter=None, save_importance=True):

    # Load and filter
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2) # convert run to str
    df = df.dropna(subset=["BDI"])
    df["label"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Filter by run
    if run_filter:
        df = df[df["run"] == run_filter]

    print(f"Participants used for classification: {len(df)} (Run={run_filter or 'All'})")

    # Features
    eeg_features = [col for col in df.columns if (
        col.startswith(('alpha_', 'beta_', 'theta_', 'delta_')) or
        col in ['faa', 'spectral_entropy']
    )]

    X = df[eeg_features].values
    y = df["label"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, f1s = [], [], []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))

    # Save metrics
    results = pd.DataFrame({
        "metric": ["accuracy", "auc", "f1"],
        "mean": [np.mean(accs), np.mean(aucs), np.mean(f1s)],
        "std": [np.std(accs), np.std(aucs), np.std(f1s)]
    })

    suffix = f"_run{run_filter}" if run_filter else ""
    results.to_csv(f"Results/classification_results{suffix}.csv", index=False)

    # Save feature importances
    if save_importance:
        clf.fit(X_scaled, y)  # final fit on all data
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": eeg_features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        imp_df.to_csv(f"Results/feature_importance{suffix}.csv", index=False)

    print(f"✅ Classification complete (Run={run_filter or 'All'}).")


def run_classification_with_smote(features_csv="features.csv", run_filter=None, save_importance=True):

    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df = df.dropna(subset=["BDI"])
    df["label"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if run_filter:
        df = df[df["run"] == run_filter]

    print(f"Running classification with SMOTE on {len(df)} participants (Run={run_filter or 'All'})")

    eeg_features = [col for col in df.columns if (
        col.startswith(('alpha_', 'beta_', 'theta_', 'delta_')) or
        col in ['faa', 'spectral_entropy']
    )]

    X = df[eeg_features].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, f1s = [], [], []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        clf = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred))

    results = pd.DataFrame({
        "metric": ["accuracy", "auc", "f1"],
        "mean": [np.mean(accs), np.mean(aucs), np.mean(f1s)],
        "std": [np.std(accs), np.std(aucs), np.std(f1s)]
    })

    suffix = f"_run{run_filter}" if run_filter else ""
    results.to_csv(f"Results/classification_results_smote{suffix}.csv", index=False)

    if save_importance:
        clf.fit(X_scaled, y)
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": eeg_features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        imp_df.to_csv(f"Results/feature_importance_smote{suffix}.csv", index=False)
   

    print(f"✅ SMOTE classification complete. Results saved to Results/classification_results_smote{suffix}.csv")

def plot_run_comparison(SMOTE=False):
    if SMOTE:
        suffix = "_smote"
    else:
        suffix = ""
    df1 = pd.read_csv(f"Results/classification_results{suffix}_run01.csv")
    df1["run"] = "Run 01"

    df2 = pd.read_csv(f"Results/classification_results{suffix}_run02.csv")
    df2["run"] = "Run 02"
    
    df = pd.concat([df1, df2], ignore_index=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="metric", y="mean", hue="run", capsize=0.2)
    plt.title("Classification Performance: Run 01 vs Run 02")
    plt.ylabel("Score")
    plt.ylim(0.4, 1.0)
    plt.legend(title="Run")
    plt.tight_layout()
    plt.savefig(f"Plots/performance_comparison{suffix}.png", dpi=300)
    plt.clf()
    print(f"📊 Comparison plot saved to: Plots/performance_comparison{suffix}.png")

# endregion


def permutation_test(features_csv="features.csv", run_filter="01", n_permutations=100):

    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df = df[df["run"] == run_filter]
    df = df.dropna(subset=["BDI"])
    df["label"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    features = [col for col in df.columns if col.startswith(('alpha_', 'beta_', 'theta_', 'delta_')) or col in ["faa", "spectral_entropy"]]
    X = df[features].values
    y = df["label"].values

    if X.shape[0] == 0:
        print(f"❌ No data for run {run_filter}")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Real model score
    real_score = cross_val_score(clf, X, y, cv=skf, scoring='accuracy').mean()

    # Permutations
    scores = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        score = cross_val_score(clf, X, y_perm, cv=skf, scoring='accuracy').mean()
        scores.append(score)

    p_value = (np.sum(np.array(scores) >= real_score) + 1) / (n_permutations + 1)

    # Save csv
    out_df = pd.DataFrame({
        "real_score": [real_score],
        "permutation_mean": [np.mean(scores)],
        "permutation_std": [np.std(scores)],
        "p_value": [p_value]
    })
    out_df.to_csv(f"Results/permutation_test_run{run_filter}.csv", index=False)

    # Save raw permutation scores
    pd.DataFrame({"score": scores}).to_csv(f"Results/permutation_scores_run{run_filter}.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, bins=20, kde=False, color='skyblue', label='Permuted scores')
    plt.axvline(real_score, color='red', linestyle='--', linewidth=2, label=f'Real accuracy: {real_score:.3f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Accuracy Distribution (Run {run_filter})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Plots/permutation_accuracy_hist_run{run_filter}.png", dpi=300)
    plt.clf()

    print(f"✅ Saved: Results/permutation_test_run{run_filter}.csv")
    print(f"📊 Plot saved: Plots/permutation_accuracy_hist_run{run_filter}.png")

   
def feature_comparison(features_csv="features.csv"):

    # Load top 15 features from both runs
    df1 = pd.read_csv("Results/feature_importance_run01.csv").sort_values("importance", ascending=False).head(15)
    df2 = pd.read_csv("Results/feature_importance_run02.csv").sort_values("importance", ascending=False).head(15)

    df1["run"] = "Run 01"
    df2["run"] = "Run 02"

    # Combine for plot and CSV
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Basic anatomical mapping for EEG channels
    def get_region(feature_name):
        for region, channels in {
            "Frontal": ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "AF3", "AF4"],
            "Central": ["Cz", "C3", "C4", "FC1", "FC2"],
            "Parietal": ["Pz", "P3", "P4", "CP1", "CP2"],
            "Occipital": ["O1", "O2", "Oz", "POz"],
            "Temporal": ["T7", "T8", "TP9", "TP10"]
        }.items():
            for ch in channels:
                if ch in feature_name:
                    return region
        if feature_name == "faa":
            return "Frontal (asymmetry)"
        if feature_name == "spectral_entropy":
            return "Global"
        return "Unknown"

    combined_df["region"] = combined_df["feature"].apply(get_region)

    # Save CSV
    combined_df.to_csv("Results/top_features_comparison_annotated.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_df, x="importance", y="feature", hue="run")
    plt.title("Top 15 EEG Features by Importance (Run 01 vs Run 02)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("Plots/top_feature_importances_comparison.png", dpi=300)
    plt.clf()

    combined_df.head(10)

def explain_with_shap(features_csv="features.csv", run_filter="01"):
    df = pd.read_csv(features_csv)
    df["run"] = df["run"].astype(str).str.zfill(2)
    df = df[df["run"] == run_filter]
    df = df.dropna(subset=["BDI"])
    df["label"] = np.where(df["BDI"] < 8, 0, np.where(df["BDI"] > 13, 1, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    features = [col for col in df.columns if col.startswith(('alpha_', 'beta_', 'theta_', 'delta_')) or col in ['faa', 'spectral_entropy']]
    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    shap_data = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(shap_data, pd.DataFrame(X_scaled, columns=features), show=False)
    plt.title(f"SHAP Summary Plot (Run {run_filter})")
    plt.tight_layout()
    plt.savefig(f"Plots/shap_summary_run{run_filter}.png", dpi=300)
    plt.close()
    print(f"✅ SHAP summary plot saved: Plots/shap_summary_run{run_filter}.png")
