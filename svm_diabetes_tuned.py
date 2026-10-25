import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
K = 5
INNER_K = 3
OUTPUT_DIR = "outputs_tuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = "pima-indians-diabetes_data.csv"
COLUMN_MAP = {
    "times_pregnant": "preg",
    "glucose_concentration": "plas",
    "diastolic_blood_pressure": "pres",
    "triceps_sf_thickness": "skin",
    "serum_insulin": "insu",
    "bmi": "mass",
    "d_pedigree_function": "pedi",
    "years_of_age": "age",
    "diabetes": "class",
}

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH).rename(columns=COLUMN_MAP)
    df["class"] = np.where(df["class"] == 1, "tested_positive", "tested_negative")
else:
    dataset = fetch_openml(name="diabetes", version=1, as_frame=True)
    df = dataset.frame.copy()

print("Shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nDescriptive statistics:")
print(df.describe().T.round(3))

target_col = "class"
feature_cols = [c for c in df.columns if c != target_col]

zero_as_missing = ["plas", "pres", "skin", "insu", "mass"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

print("\nMissing values after treating invalid zeros as missing:")
print(df[zero_as_missing].isnull().sum())

X = df[feature_cols]
y = (df[target_col] == "tested_positive").astype(int)

print("\nClass distribution:")
print(y.value_counts().rename({0: "No diabetes", 1: "Diabetes"}))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

fig, ax = plt.subplots(figsize=(7, 5))
corr = df[feature_cols].corr()
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=45, ha="right")
ax.set_yticklabels(feature_cols)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=200)
plt.close(fig)


def build_pipeline(kernel, **params):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, random_state=RANDOM_STATE, max_iter=200000, **params)),
        ]
    )


baseline_params = {
    "linear": {"C": 1.0},
    "rbf": {"C": 1.0, "gamma": "scale"},
    "poly": {"C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
    "sigmoid": {"C": 1.0, "gamma": "scale", "coef0": 0.0},
}

param_grids = {
    "linear": {
        "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "svm__class_weight": [None, "balanced"],
    },
    "rbf": {
        "svm__C": [0.1, 1, 10, 100, 1000],
        "svm__gamma": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
        "svm__class_weight": [None, "balanced"],
    },
    "poly": {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": [0.01, 0.1],
        "svm__degree": [2, 3],
        "svm__coef0": [0, 1],
        "svm__class_weight": [None, "balanced"],
    },
    "sigmoid": {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": [0.0001, 0.001, 0.01, 0.1, 1],
        "svm__coef0": [0, 1],
        "svm__class_weight": [None, "balanced"],
    },
}

scoring = {"accuracy": "accuracy", "f1": "f1", "precision": "precision", "recall": "recall"}
inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=RANDOM_STATE)
outer_cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)


def evaluate(model, name, tag):
    y_pred = model.predict(X_test)
    return {
        "Kernel": name,
        "Setting": tag,
        "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }, y_pred


rows = []
tuned_models = {}
tuned_confusions = {}
baseline_confusions = {}
best_params_table = {}
grid_results = {}

for name in baseline_params:
    base_model = build_pipeline(name, **baseline_params[name])
    base_model.fit(X_train, y_train)
    base_row, base_pred = evaluate(base_model, name, "Default")
    rows.append(base_row)
    baseline_confusions[name] = confusion_matrix(y_test, base_pred)

    grid = GridSearchCV(
        build_pipeline(name),
        param_grids[name],
        scoring=scoring,
        refit="f1",
        cv=inner_cv,
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    tuned_models[name] = grid.best_estimator_
    best_params_table[name] = {
        k.replace("svm__", ""): v for k, v in grid.best_params_.items()
    }
    grid_results[name] = pd.DataFrame(grid.cv_results_)

    tuned_row, tuned_pred = evaluate(grid.best_estimator_, name, "Tuned")
    rows.append(tuned_row)
    tuned_confusions[name] = confusion_matrix(y_test, tuned_pred)

    print("\n" + "=" * 70)
    print("Kernel:", name)
    print("Best params:", best_params_table[name])
    print("Best inner-CV F1:", round(grid.best_score_, 4))
    print("Confusion matrix (tuned):\n", tuned_confusions[name])
    print(
        classification_report(
            y_test,
            tuned_pred,
            target_names=["No diabetes", "Diabetes"],
            zero_division=0,
        )
    )

results_df = pd.DataFrame(rows).set_index(["Kernel", "Setting"])
print("\nTest set comparison (default vs tuned):")
print(results_df.round(4))
results_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "kernel_comparison_default_vs_tuned.csv"))

best_params_df = pd.DataFrame(best_params_table).T
print("\nBest hyperparameters per kernel:")
print(best_params_df)
best_params_df.to_csv(os.path.join(OUTPUT_DIR, "best_hyperparameters.csv"))

fig, axes = plt.subplots(
    2, len(tuned_confusions), figsize=(5 * len(tuned_confusions), 9)
)
labels = ["No diabetes", "Diabetes"]
for col, name in enumerate(tuned_confusions):
    ConfusionMatrixDisplay(baseline_confusions[name], display_labels=labels).plot(
        ax=axes[0, col], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0, col].set_title(f"{name} - Default")
    ConfusionMatrixDisplay(tuned_confusions[name], display_labels=labels).plot(
        ax=axes[1, col], cmap="Greens", colorbar=False, values_format="d"
    )
    axes[1, col].set_title(f"{name} - Tuned")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices_default_vs_tuned.png"), dpi=200)
plt.show()

for name, cm in tuned_confusions.items():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Greens", values_format="d"
    )
    ax.set_title(f"Confusion Matrix (Tuned) - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_tuned_{name}.png"), dpi=200)
    plt.close(fig)

metrics_to_plot = ["Test Accuracy", "Precision", "Recall", "F1-score"]
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, tag in zip(axes, ["Default", "Tuned"]):
    results_df.xs(tag, level="Setting")[metrics_to_plot].plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_title(f"Test Metrics - {tag}")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_default_vs_tuned.png"), dpi=200)
plt.show()

rbf_df = grid_results["rbf"].copy()
rbf_df["C"] = rbf_df["param_svm__C"].astype(float)
rbf_df["gamma"] = rbf_df["param_svm__gamma"].astype(float)
heat = (
    rbf_df.groupby(["C", "gamma"])["mean_test_f1"]
    .max()
    .unstack("gamma")
    .sort_index()
)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heat.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels([str(g) for g in heat.columns])
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels([str(c) for c in heat.index])
ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_title("RBF Kernel GridSearch - Mean Inner-CV F1")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rbf_gridsearch_heatmap.png"), dpi=200)
plt.show()

cv_summary = []
fold_accuracy = {}
fold_f1 = {}

for name in param_grids:
    nested_model = GridSearchCV(
        build_pipeline(name),
        param_grids[name],
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
    )
    scores = cross_validate(
        nested_model, X, y, cv=outer_cv, scoring=scoring, n_jobs=1
    )

    fold_accuracy[name] = scores["test_accuracy"]
    fold_f1[name] = scores["test_f1"]

    print("\n" + "=" * 70)
    print(f"Nested {K}-Fold Cross-Validation - kernel: {name}")
    for i in range(K):
        print(
            f"Fold {i + 1}: Accuracy = {scores['test_accuracy'][i]:.4f} | "
            f"F1 = {scores['test_f1'][i]:.4f}"
        )
    print(
        f"Mean Accuracy = {scores['test_accuracy'].mean():.4f} "
        f"(+/- {scores['test_accuracy'].std():.4f})"
    )
    print(
        f"Mean F1-score = {scores['test_f1'].mean():.4f} "
        f"(+/- {scores['test_f1'].std():.4f})"
    )

    cv_summary.append(
        {
            "Kernel": name,
            "Mean Accuracy": scores["test_accuracy"].mean(),
            "Std Accuracy": scores["test_accuracy"].std(),
            "Mean F1": scores["test_f1"].mean(),
            "Std F1": scores["test_f1"].std(),
            "Mean Precision": scores["test_precision"].mean(),
            "Mean Recall": scores["test_recall"].mean(),
        }
    )

cv_df = pd.DataFrame(cv_summary).set_index("Kernel")
print("\nNested cross-validation summary:")
print(cv_df.round(4))
cv_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "nested_cross_validation_summary.csv"))

fold_index = [f"Fold {i + 1}" for i in range(K)]
fold_acc_df = pd.DataFrame(fold_accuracy, index=fold_index)
fold_f1_df = pd.DataFrame(fold_f1, index=fold_index)
print("\nPer-fold accuracy:")
print(fold_acc_df.round(4))
print("\nPer-fold F1-score:")
print(fold_f1_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fold_acc_df.plot(kind="bar", ax=axes[0], rot=0)
axes[0].set_ylim(0, 1)
axes[0].set_title("Accuracy per Fold (Nested CV)")
axes[0].set_ylabel("Accuracy")
fold_f1_df.plot(kind="bar", ax=axes[1], rot=0)
axes[1].set_ylim(0, 1)
axes[1].set_title("F1-score per Fold (Nested CV)")
axes[1].set_ylabel("F1-score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_folds.png"), dpi=200)
plt.show()

names = list(fold_accuracy.keys())
plt.figure(figsize=(8, 5))
plt.boxplot([fold_accuracy[n] for n in names])
plt.xticks(range(1, len(names) + 1), names)
plt.ylabel("Accuracy")
plt.title(f"Nested {K}-Fold CV Accuracy Distribution per Kernel")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_boxplot.png"), dpi=200)
plt.show()

best_kernel = cv_df["Mean F1"].idxmax()
print("\nBest kernel by mean nested-CV F1-score:", best_kernel)
print("Mean nested-CV Accuracy:", round(cv_df.loc[best_kernel, "Mean Accuracy"], 4))
print("Mean nested-CV F1-score:", round(cv_df.loc[best_kernel, "Mean F1"], 4))
print("Best hyperparameters on the training split:", best_params_table[best_kernel])
