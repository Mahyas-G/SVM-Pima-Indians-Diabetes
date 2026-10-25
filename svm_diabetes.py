import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
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
print("\nMissing values per column:")
print(df.isnull().sum())

target_col = "class"
feature_cols = [c for c in df.columns if c != target_col]

zero_as_missing = ["plas", "pres", "skin", "insu", "mass"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

print("\nMissing values after treating zeros as missing:")
print(df[zero_as_missing].isnull().sum())

X = df[feature_cols]
y = (df[target_col] == "tested_positive").astype(int)

print("\nClass distribution:")
print(y.value_counts().rename({0: "No diabetes", 1: "Diabetes"}))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

kernel_configs = {
    "linear": {"kernel": "linear", "C": 1.0},
    "rbf": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    "poly": {"kernel": "poly", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
    "sigmoid": {"kernel": "sigmoid", "C": 1.0, "gamma": "scale", "coef0": 0.0},
}


def build_pipeline(params):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(random_state=RANDOM_STATE, **params)),
        ]
    )


results = []
confusion_matrices = {}

for name, params in kernel_configs.items():
    model = build_pipeline(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    train_acc = accuracy_score(y_train, model.predict(X_train))

    results.append(
        {
            "Kernel": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
        }
    )

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    print("\n" + "=" * 60)
    print("Kernel:", name, "| Params:", params)
    print("Confusion matrix:\n", cm)
    print(
        classification_report(
            y_test, y_pred, target_names=["No diabetes", "Diabetes"], zero_division=0
        )
    )

results_df = pd.DataFrame(results).set_index("Kernel")
print("\nTest set comparison:")
print(results_df.round(4))
results_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "kernel_comparison.csv"))

fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(5 * len(confusion_matrices), 4.5))
for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    disp = ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices_all.png"), dpi=200)
plt.show()

for name, cm in confusion_matrices.items():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"), dpi=200)
    plt.close(fig)

ax = results_df[["Test Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(10, 5), rot=0
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("SVM Kernel Comparison on Test Set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kernel_comparison_bar.png"), dpi=200)
plt.show()

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
scoring = {"accuracy": "accuracy", "f1": "f1", "precision": "precision", "recall": "recall"}

cv_summary = []
cv_fold_accuracy = {}
cv_fold_f1 = {}

for name, params in kernel_configs.items():
    model = build_pipeline(params)
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    cv_fold_accuracy[name] = scores["test_accuracy"]
    cv_fold_f1[name] = scores["test_f1"]

    print("\n" + "=" * 60)
    print(f"{K}-Fold Cross-Validation - kernel: {name}")
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
print("\nCross-validation summary:")
print(cv_df.round(4))
cv_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "cross_validation_summary.csv"))

fold_acc_df = pd.DataFrame(cv_fold_accuracy, index=[f"Fold {i + 1}" for i in range(K)])
fold_f1_df = pd.DataFrame(cv_fold_f1, index=[f"Fold {i + 1}" for i in range(K)])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fold_acc_df.plot(kind="bar", ax=axes[0], rot=0)
axes[0].set_ylim(0, 1)
axes[0].set_title("Accuracy per Fold")
axes[0].set_ylabel("Accuracy")
fold_f1_df.plot(kind="bar", ax=axes[1], rot=0)
axes[1].set_ylim(0, 1)
axes[1].set_title("F1-score per Fold")
axes[1].set_ylabel("F1-score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_folds.png"), dpi=200)
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([cv_fold_accuracy[k] for k in kernel_configs])
plt.xticks(range(1, len(kernel_configs) + 1), list(kernel_configs.keys()))
plt.ylabel("Accuracy")
plt.title(f"{K}-Fold CV Accuracy Distribution per Kernel")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_boxplot.png"), dpi=200)
plt.show()

best_kernel = cv_df["Mean Accuracy"].idxmax()
print("\nBest kernel by mean CV accuracy:", best_kernel)
print("Mean CV accuracy:", round(cv_df.loc[best_kernel, "Mean Accuracy"], 4))
print("Mean CV F1-score:", round(cv_df.loc[best_kernel, "Mean F1"], 4))
