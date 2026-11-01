import sys
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")


def load_data(csv_path=None):
    print("[1/6] Loading recruitment data...")

    if csv_path:
        resolved_path = Path(csv_path)
    else:
        resolved_path = None
        raise FileNotFoundError(
            "No CSV file was found. Please provide a path to your dataset, for example: python project_iti.py your_file.csv"
        )

    df = pd.read_csv(resolved_path)

    return df

def preprocess_numeric(df):
     work_df = df.copy()
     

     #convert experience column to numeric, handling special cases like '>20' and '<1'
     work_df["experience"] = pd.to_numeric(
             work_df["experience"].astype(str).str.replace(">", "", regex=False).str.replace("<", "", regex=False),
             errors="coerce",
         )
        
     #map education levels to numerical values for interaction features
     education_rank = {"Primary School": 0, "High School": 1, "Graduate": 2, "Masters": 3, "PhD": 4}
     work_df["education_level"] = work_df["education_level"].map(education_rank).fillna(2)


     #create interaction features between education and training hours    
     work_df["education_training_interaction"] = work_df["education_level"] * work_df["training_hours"].fillna(0)


     #create a binary feature for relevant experience    
     work_df["relevent_experience_binary"] = (
        work_df["relevent_experience"].fillna("No relevant experience").str.contains("Has", case=False, na=False).astype(int)
         )


     #create interaction features between education, training, and relevant experience
     work_df["education_experience_interaction"] = work_df["education_level"] * work_df["relevent_experience_binary"].fillna(5.0)
     work_df["training_experience_interaction"] = work_df["training_hours"].fillna(0) * work_df["relevent_experience_binary"].fillna(5.0)


     #convert company_size to numeric handling special cases like '10/49', '50-99', and '<10'
     work_df["company_size_missing"] = work_df["company_size"].isna().astype(int)
     work_df["company_size"] = (
         work_df["company_size"]
         .astype(str)
         .str.replace("10/49", "29.5")
         .str.replace("50-99", "74.5")
         .str.replace("100-500", "300")
         .str.replace("500-999", "749.5")
         .str.replace("1000-5000", "3000")
         .str.replace("5000-9999", "7499.5")
         .str.replace("10000+", "15000")
         .str.replace("<10", "5")
     )
     work_df["company_size"] = pd.to_numeric(work_df["company_size"], errors="coerce")

     # extra numeric signals that often help with recruitment outcomes
     work_df["last_new_job_num"] = (
         work_df["last_new_job"]
         .astype(str)
         .str.replace("never", "0", regex=False)
         .str.replace(">", "", regex=False)
         .str.replace("<", "", regex=False)
         .str.strip()
     )
     work_df["last_new_job_num"] = pd.to_numeric(work_df["last_new_job_num"], errors="coerce")
     work_df["last_new_job_missing"] = work_df["last_new_job"].isna().astype(int)
    

     return work_df


def preprocess_data(df):
    print("[2/6] Preprocessing features and preparing training data...")
    work_df = df.copy()

    work_df = preprocess_numeric(work_df)

   
    
    
    # Handle rare categories in company_type 
    
    work_df["company_type_is_pvt_ltd"] = work_df["company_type"].fillna("Other").eq("Pvt Ltd").astype(int)

    work_df["company_type"] = work_df["company_type"].fillna("Other").where(
        work_df["company_type"].fillna("Other").eq("Pvt Ltd"), "Other"
    )


    #handle rare categories in city and major_discipline by grouping them into 'Other'
    for column, threshold in [("city", 20), ("major_discipline", 10)]:
        counts = work_df[column].value_counts(dropna=False)
        rare_values = counts[counts < threshold].index
        if len(rare_values) > 0:
            work_df[column] = work_df[column].where(~work_df[column].isin(rare_values), "Other")

    work_df["city"] = work_df["city"].fillna("Other")
    work_df["major_discipline"] = work_df["major_discipline"].fillna("Other")
    
    #keep track of candidate IDs for ranking output
    candidate_ids = work_df["enrollee_id"]

    
    categorical_features = [
        "city",
        "gender",
        "enrolled_university",
        "major_discipline",
        "company_type",
        "last_new_job",
        "relevent_experience",
    ]
    numerical_features = [
        "training_hours",
        "experience",
        "education_level",
        "education_experience_interaction",
        "education_training_interaction",
        "training_experience_interaction",
        "city_development_index",
        "relevent_experience_binary",
        "company_size_missing",
        "company_type_is_pvt_ltd",
        "company_size",
        "last_new_job_num",
        "last_new_job_missing",
        
    ]

    X = work_df[categorical_features + numerical_features]
    y = work_df["target"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    selected_features = list(X.columns)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    selected_categorical_features = [col for col in categorical_features if col in selected_features]
    selected_numerical_features = [col for col in numerical_features if col in selected_features]

    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                selected_categorical_features,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                selected_numerical_features,
            ),
        ],
        remainder="drop",
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "candidate_ids": candidate_ids,
        "preprocessor": preprocessor,
        "feature_frame": work_df,
        "selected_features": selected_features,
    }


def train_models(preprocessed_data):
    
    print("[3/6] Training and tuning classification models...")
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    preprocessor = preprocessed_data["preprocessor"]

    #logistic regression
    log_reg_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=4000, random_state=42)),
        ]
    )
    log_grid = {"model__C": [20]}
    log_search = GridSearchCV(
        log_reg_pipeline,
        param_grid=log_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    log_search.fit(X_train, y_train)

    #random forest
    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_estimators=250, max_depth=None, min_samples_leaf=2)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200],
        "model__max_depth": [None],
        "model__min_samples_leaf": [2],
        
    }
    rf_search = GridSearchCV(
        rf_pipeline,
        param_grid=rf_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)

    #gradiant boosting

    gb_pipeline=Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
        ]
    )
    gb_grid={
        "model__n_estimators": [30],
        "model__max_depth": [4],
    }
    gb_search=GridSearchCV(
        gb_pipeline,
        param_grid=gb_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    gb_search.fit(X_train,y_train)
    #print("""look here now now now nown onwonowno""",log_search.best_params_,rf_search.best_params_,gb_search.best_params_)
    return {"logistic_regression": log_search.best_estimator_, "random_forest": rf_search.best_estimator_,"gradiant_boosting":gb_search.best_estimator_}


def evaluate_models(models, preprocessed_data):
    
    print("[4/6] Evaluating models on the test set...")
    X_test = preprocessed_data["X_test"]
    y_test = preprocessed_data["y_test"]


    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        results[name] = metrics

        print(f"\n{name} Summary")
        print("-" * 40)
        print(pd.DataFrame([metrics], index=[name]).round(4))
        print("\nClassification Report")
        print(classification_report(y_test, y_pred, target_names=["Reject", "Advance"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Reject", "Advance"], yticklabels=["Reject", "Advance"])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_').lower()}_confusion_matrix.png", dpi=300)
        plt.close()

    metrics_df = pd.DataFrame(results).T.round(4)
    print("\nComparative Performance")
    print(metrics_df)

    return metrics_df


def rank_candidates(model, data, top_n=10):
    """Return the top candidates ranked by predicted probability of advancing."""
    print(f"[5/6] Ranking the top {top_n} candidates...")
    ranked = data.copy()
    ranked["predicted_probability"] = model.predict_proba(ranked)[:, 1]
    ranked = ranked.sort_values("predicted_probability", ascending=False).head(top_n).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked = ranked[["rank", "enrollee_id", "city", "education_level", "experience", "company_type", "training_hours", "relevent_experience", "predicted_probability"]]

    print("\nTop Recommended Candidates")
    print(ranked.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return ranked


def generate_simple_dashboard(models, metrics_df, top_candidates, output_html="simple_dashboard.html", output_png="simple_dashboard.png"):
    """Create a simple, readable dashboard summary for the recruitment project."""
    print("[6/6] Generating simple dashboard summary...")

    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradiant_boosting": "Gradient Boosting",
    }

    perf_df = metrics_df[["accuracy", "f1", "roc_auc"]].copy()
    perf_df.index = [display_names.get(name, name) for name in perf_df.index]

    rf_model = models["random_forest"]
    rf_pipeline = rf_model
    rf_feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
    rf_importance = pd.Series(rf_pipeline.named_steps["model"].feature_importances_, index=rf_feature_names)
    top_features = rf_importance.sort_values(ascending=False).head(8)

    top_candidates_summary = top_candidates.head(5).copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    perf_df.plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518", "#54A24B"], width=0.8)
    axes[0].set_title("Model Performance")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left")
    axes[0].tick_params(axis="x", rotation=20)

    top_features.sort_values(ascending=True).plot(kind="barh", ax=axes[1], color="#2F80ED")
    axes[1].set_title("Top Features")
    axes[1].set_xlabel("Importance")
    axes[1].invert_yaxis()

    candidate_ids = [str(value) for value in top_candidates_summary["enrollee_id"]]
    candidate_scores = top_candidates_summary["predicted_probability"]
    axes[2].bar(candidate_ids, candidate_scores, color="#27AE60")
    axes[2].set_title("Top Candidate Scores")
    axes[2].set_ylabel("Advance Probability")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0, 1.0)

    fig.suptitle("Simple Recruitment Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    best_model = metrics_df["f1"].idxmax()
    top_feature_names = ", ".join(top_features.index[:5].astype(str))
    best_candidate = top_candidates.iloc[0]

    metrics_rows = "".join(
        f"<tr><td>{escape(display_names.get(name, name))}</td><td>{metrics_df.loc[name, 'accuracy']:.3f}</td><td>{metrics_df.loc[name, 'f1']:.3f}</td><td>{metrics_df.loc[name, 'roc_auc']:.3f}</td></tr>"
        for name in metrics_df.index
    )

    candidate_table_rows = "".join(
        "<tr>"
        f"<td>{row.rank}</td>"
        f"<td>{escape(str(row.enrollee_id))}</td>"
        f"<td>{escape(str(row.city))}</td>"
        f"<td>{escape(str(row.education_level))}</td>"
        f"<td>{escape(str(row.experience))}</td>"
        f"<td>{escape(str(row.company_type))}</td>"
        f"<td>{escape(str(row.training_hours))}</td>"
        f"<td>{escape(str(row.relevent_experience))}</td>"
        f"<td>{row.predicted_probability:.3f}</td>"
        "</tr>"
        for row in top_candidates.head(10).itertuples(index=False)
    )

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Smart Recruitment Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f8fb; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
    th {{ background: #eef2ff; }}
    img {{ width: 100%; max-width: 1000px; border-radius: 8px; margin-top: 16px; }}
  </style>
</head>
<body>
  <h1>Smart Recruitment Assistant Dashboard</h1>
  
  <div class=\"cards\">
    <div class=\"card\"><strong>Best Model by F1</strong><br />{escape(display_names.get(best_model, best_model))}</div>
    <div class=\"card\"><strong>Top Candidate</strong><br />ID {best_candidate['enrollee_id']} ({best_candidate['predicted_probability']:.3f})</div>
    <div class=\"card\"><strong>Top Feature</strong><br />{escape(top_features.index[0])}</div>
  </div>

  <h2>Model Metrics</h2>
  <table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC AUC</th></tr></thead>
    <tbody>{metrics_rows}</tbody>
  </table>

  <h2>Visual Summary</h2>
  <img src=\"{output_png}\" alt=\"Recruitment dashboard chart\" />

  <h2>Top 10 Candidates</h2>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Enrollee ID</th>
        <th>City</th>
        <th>Education Level</th>
        <th>Experience</th>
        <th>Company Type</th>
        <th>Training Hours</th>
        <th>Relevant Experience</th>
        <th>Predicted Probability</th>
      </tr>
    </thead>
    <tbody>{candidate_table_rows}</tbody>
  </table>

  
</body>
</html>
"""

    Path(output_html).write_text(html_content, encoding="utf-8")
    print(f"Saved dashboard files: {output_png}, {output_html}")


def generate_bi_dashboard(models, metrics_df, feature_frame, X_test, y_test):
    """Create and save business intelligence charts for recruitment insights."""
    print("[6/6] Generating BI dashboard charts...")

    rf_model = models["random_forest"]
    lr_model = models["logistic_regression"]
    gb_model=models["gradiant_boosting"]


    rf_pipeline = rf_model
    lr_pipeline = lr_model
    gb_pipeline=gb_model

    rf_feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
    lr_feature_names = lr_pipeline.named_steps["preprocess"].get_feature_names_out()
    gb_feature_names=gb_pipeline.named_steps["preprocess"].get_feature_names_out()

    rf_importance = pd.Series(rf_pipeline.named_steps["model"].feature_importances_, index=rf_feature_names)
    lr_coefficients = pd.Series(lr_pipeline.named_steps["model"].coef_[0], index=lr_feature_names)
    gb_importance=pd.Series(gb_pipeline.named_steps["model"].feature_importances_, index=gb_feature_names)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    rf_importance.sort_values(ascending=False).head(12).plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].set_xlabel("Importance")
    axes[0].invert_yaxis()

    lr_coefficients.abs().sort_values(ascending=False).head(12).plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Logistic Regression | Absolute Coefficients")
    axes[1].set_xlabel("Absolute Coefficient")
    axes[1].invert_yaxis()

    gb_importance.sort_values(ascending=False).head(12).plot(kind="barh", ax=axes[2], color="forestgreen")
    axes[2].set_title("Gradient Boosting Feature Importance")
    axes[2].set_xlabel("Importance")
    axes[2].invert_yaxis()

    fig.suptitle("Feature Importance Dashboard", y=1.02)
    plt.tight_layout()
    plt.savefig("feature_importance_dashboard.png", dpi=300)
    plt.close()


    
    print("Saved charts: feature_importance_dashboard.png, candidate_profile_trends.png, model_performance_comparison.png")


def main(csv_path=None):
    
    print("=" * 70)
    print("SMART RECRUITMENT ASSISTANT")
    print("=" * 70)
    

    raw_df = load_data(csv_path)
    prepared = preprocess_data(raw_df)
    models = train_models(prepared)
    
    metrics_df = evaluate_models(models, prepared)

    top_candidates = rank_candidates(models["random_forest"], prepared["feature_frame"].copy())
    #generate_simple_dashboard(models, metrics_df, top_candidates)
    generate_bi_dashboard(models, metrics_df, prepared["feature_frame"], prepared["X_test"], prepared["y_test"])

    print("\nPipeline complete. Charts and ranking output saved successfully.")
    return top_candidates


if __name__ == "__main__":
    csv_arg = "aug_train.csv"  
    main(csv_arg)
