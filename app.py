from flask import Flask, render_template
import pandas as pd
from analytics import get_analytics

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():

    # Load the RAW dataset
    df = pd.read_csv("gym_members_exercise_tracking_synthetic_data.csv")

    # Dataset Information
    dataset_name = "Gym Members Exercise Tracking Dataset"
    dataset_source = "Kaggle"

    total_rows = df.shape[0]
    total_columns = df.shape[1]

    # First 10 rows
    table = df.head(10).to_html(
        classes="dataset-table",
        index=False
    )

    return render_template(
        "dashboard.html",
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        rows=total_rows,
        columns=total_columns,
        table=table
    )

@app.route("/datacleaning")
def datacleaning():

    cleaned_df = pd.read_csv("fitness_tracker_cleaned.csv")

    rows = cleaned_df.shape[0]
    columns = cleaned_df.shape[1]

    missing_values = cleaned_df.isnull().sum().sum()
    duplicates = cleaned_df.duplicated().sum()

    cleaned_table = cleaned_df.head(10).to_html(
        classes="dataset-table",
        index=False
    )

    return render_template(
        "datacleaning.html",
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        duplicates=duplicates,
        cleaned_table=cleaned_table
    )

@app.route("/analytics")
def analytics():

    analytics = get_analytics()

    return render_template(
        "analytics.html",
        analytics=analytics
    )

@app.route("/visuals")
def visuals():
    return render_template("visuals.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

if __name__ == "__main__":
    app.run(debug=True)