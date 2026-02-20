import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical-data",
        type=Path,
        default=Path("cancer clinic/CFB-GBM_clinical_data_v02_20260129 (1).tsv"),
    )
    parser.add_argument(
        "--treatment-data",
        type=Path,
        default=Path("cancer clinic/CFB-GBM_treatment_data_v02_20260129.tsv"),
    )
    parser.add_argument(
        "--imaging-data",
        type=Path,
        default=Path("cancer clinic/CFB-GBM_treatment_imaging_availability_v02_20260129.tsv"),
    )
    parser.add_argument("--target", default="who_performance_status")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("cancer clinic/who_performance_status_nn.joblib"),
    )
    args = parser.parse_args()

    clinical = pd.read_csv(args.clinical_data, sep="\t")
    treatment = pd.read_csv(args.treatment_data, sep="\t")
    imaging = pd.read_csv(args.imaging_data, sep="\t")

    if (
        "id_patient" not in clinical.columns
        or "id_patient" not in treatment.columns
        or "id_patient" not in imaging.columns
    ):
        raise ValueError("All files must include id_patient")

    # Keep one row per patient from imaging table before merge.
    imaging = imaging.sort_values("id_patient").drop_duplicates(subset=["id_patient"], keep="first")

    df = clinical.merge(treatment, on="id_patient", how="left")
    df = df.merge(imaging, on="id_patient", how="left")

    if args.target not in clinical.columns:
        raise ValueError(f"Missing target column: {args.target}")

    df = df.dropna(subset=[args.target]).copy()
    X = df.drop(columns=[args.target, "id_patient"], errors="ignore")
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", num_pipe, selector(dtype_exclude="object")),
            ("cat", cat_pipe, selector(dtype_include="object")),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "nn",
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    learning_rate_init=1e-3,
                    max_iter=500,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"rows: {len(df)}")
    print(f"features: {X.shape[1]}")
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"saved: {args.model_out}")


if __name__ == "__main__":
    main()
