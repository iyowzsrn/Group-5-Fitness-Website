import pandas as pd
from scipy import stats

CSV_PATH = "fitness_tracker_cleaned.csv"


def get_analytics():

    df = pd.read_csv(CSV_PATH)

    # Experience labels
    exp_map = {
        1.0: "Beginner",
        2.0: "Intermediate",
        3.0: "Advanced"
    }

    df["experience_label"] = df["experience_level"].map(exp_map)

    # ---------- Basic Statistics ----------

    total_users = len(df)

    avg_bmi = round(df["bmi"].mean(), 2)
    avg_fat = round(df["fat_percentage"].mean(), 2)
    avg_resting = round(df["resting_bpm"].mean(), 2)
    avg_avg_bpm = round(df["avg_bpm"].mean(), 2)
    avg_max_bpm = round(df["max_bpm"].mean(), 2)
    avg_calories = round(df["calories_burned"].mean(), 2)
    avg_session = round(df["session_duration_hours"].mean(), 2)

    common_workout = df["workout_type"].mode()[0]
    common_experience = df["experience_label"].mode()[0]

    # ---------- Pearson Correlation ----------

    r_bmi_fat, p_bmi_fat = stats.pearsonr(
        df["bmi"],
        df["fat_percentage"]
    )

    r_rest_cal, p_rest_cal = stats.pearsonr(
        df["resting_bpm"],
        df["calories_burned"]
    )

    # ---------- Chi Square ----------

    ct = pd.crosstab(df["workout_type"], df["gender"])

    chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

    # ---------- Kruskal Wallis ----------

    EXP_ORDER = [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]

    groups_bmi = [
        df[df["experience_label"] == lvl]["bmi"].dropna()
        for lvl in EXP_ORDER
    ]

    h_bmi, p_bmi = stats.kruskal(*groups_bmi)

    # ---------- Interpretations ----------

    if p_chi < 0.05:
        chi_interpretation = (
            "There is a significant association between Workout Type and Gender."
        )
    else:
        chi_interpretation = (
            "There is no significant association between Workout Type and Gender."
        )

    if p_bmi < 0.05:
        kruskal_interpretation = (
            "BMI differs significantly across experience levels."
        )
    else:
        kruskal_interpretation = (
            "BMI does not differ significantly across experience levels."
        )

    def correlation_strength(r):

        r = abs(r)

        if r >= 0.80:
            return "Very Strong"

        elif r >= 0.60:
            return "Strong"

        elif r >= 0.40:
            return "Moderate"

        elif r >= 0.20:
            return "Weak"

        else:
            return "Very Weak"

    corr_strength = correlation_strength(r_bmi_fat)

    if p_bmi_fat < 0.05:
        correlation_interpretation = (
            f"There is a {corr_strength.lower()} positive relationship "
            f"between BMI and Body Fat Percentage."
        )
    else:
        correlation_interpretation = (
            "No significant relationship was found between BMI and Body Fat Percentage."
        )

    # ---------- Significant Tests ----------

    significant_tests = 0

    if p_chi < 0.05:
        significant_tests += 1

    if p_bmi < 0.05:
        significant_tests += 1

    if p_bmi_fat < 0.05:
        significant_tests += 1

    # ---------- Strongest Relationship ----------

    if abs(r_bmi_fat) > abs(r_rest_cal):

        strongest_relationship = (
            "BMI and Body Fat Percentage"
        )

        strongest_r = round(r_bmi_fat, 4)

    else:

        strongest_relationship = (
            "Resting BPM and Calories Burned"
        )

        strongest_r = round(r_rest_cal, 4)

    # ---------- Dataset Information ----------

    valid_records = len(df)

    # ---------- Key Findings ----------

    finding_1 = (
        f"The most common workout type among participants is {common_workout}."
    )

    finding_2 = (
        f"The average BMI of participants is {avg_bmi}."
    )

    finding_3 = (
        f"The strongest relationship observed was between "
        f"{strongest_relationship} (r = {strongest_r})."
    )

    # ---------- Return everything ----------

    bmi_mean = round(df["bmi"].mean(), 2)
    bmi_median = round(df["bmi"].median(), 2)
    bmi_mode = round(df["bmi"].mode()[0], 2)
    bmi_sd = round(df["bmi"].std(), 2)
    bmi_cv = round((bmi_sd / bmi_mean) * 100, 2)

    calories_mean = round(df["calories_burned"].mean(), 2)
    calories_median = round(df["calories_burned"].median(), 2)
    calories_mode = round(df["calories_burned"].mode()[0], 2)
    calories_sd = round(df["calories_burned"].std(), 2)
    calories_cv = round((calories_sd / calories_mean) * 100, 2)

    resting_mean = round(df["resting_bpm"].mean(), 2)
    resting_median = round(df["resting_bpm"].median(), 2)
    resting_mode = round(df["resting_bpm"].mode()[0], 2)
    resting_sd = round(df["resting_bpm"].std(), 2)
    resting_cv = round((resting_sd / resting_mean) * 100, 2)

    # BMI

    bmi_min = round(df["bmi"].min(),2)
    bmi_max = round(df["bmi"].max(),2)
    bmi_range = round(bmi_max - bmi_min,2)
    bmi_p25 = round(df["bmi"].quantile(0.25),2)
    bmi_p75 = round(df["bmi"].quantile(0.75),2)
    bmi_skew = round(df["bmi"].skew(),2)
    bmi_kurtosis = round(df["bmi"].kurtosis(),2)

    # Calories

    calories_min = round(df["calories_burned"].min(),2)
    calories_max = round(df["calories_burned"].max(),2)
    calories_range = round(calories_max - calories_min,2)
    calories_p25 = round(df["calories_burned"].quantile(0.25),2)
    calories_p75 = round(df["calories_burned"].quantile(0.75),2)
    calories_skew = round(df["calories_burned"].skew(),2)
    calories_kurtosis = round(df["calories_burned"].kurtosis(),2)

    # Resting BPM

    resting_min = round(df["resting_bpm"].min(),2)
    resting_max = round(df["resting_bpm"].max(),2)
    resting_range = round(resting_max - resting_min,2)
    resting_p25 = round(df["resting_bpm"].quantile(0.25),2)
    resting_p75 = round(df["resting_bpm"].quantile(0.75),2)
    resting_skew = round(df["resting_bpm"].skew(),2)
    resting_kurtosis = round(df["resting_bpm"].kurtosis(),2)


    return {

        "total_users": total_users,

        "avg_bmi": avg_bmi,
        "avg_fat": avg_fat,
        "avg_resting": avg_resting,
        "avg_avg_bpm": avg_avg_bpm,
        "avg_max_bpm": avg_max_bpm,
        "avg_calories": avg_calories,
        "avg_session": avg_session,

        "common_workout": common_workout,
        "common_experience": common_experience,

        "bmi_fat_r": round(r_bmi_fat, 4),
        "bmi_fat_p": round(p_bmi_fat, 4),

        "rest_cal_r": round(r_rest_cal, 4),
        "rest_cal_p": round(p_rest_cal, 4),

        "chi_square": round(chi2, 2),
        "chi_p": round(p_chi, 4),

        "kruskal_h": round(h_bmi, 2),
        "kruskal_p": round(p_bmi, 4),

        "chi_interpretation": chi_interpretation,
        "kruskal_interpretation": kruskal_interpretation,

        "correlation_strength": corr_strength,
        "correlation_interpretation": correlation_interpretation,

        "significant_tests": significant_tests,

        "strongest_relationship": strongest_relationship,
        "strongest_r": strongest_r,

        "valid_records": valid_records,

        "bmi_mean": bmi_mean,
        "bmi_median": bmi_median,
        "bmi_mode": bmi_mode,
        "bmi_sd": bmi_sd,
        "bmi_cv": bmi_cv,

        "calories_mean": calories_mean,
        "calories_median": calories_median,
        "calories_mode": calories_mode,
        "calories_sd": calories_sd,
        "calories_cv": calories_cv,

        "resting_mean": resting_mean,
        "resting_median": resting_median,
        "resting_mode": resting_mode,
        "resting_sd": resting_sd,
        "resting_cv": resting_cv,

        "bmi_min": bmi_min,
        "bmi_max": bmi_max,
        "bmi_range": bmi_range,
        "bmi_p25": bmi_p25,
        "bmi_p75": bmi_p75,
        "bmi_skew": bmi_skew,
        "bmi_kurtosis": bmi_kurtosis,

        "calories_min": calories_min,
        "calories_max": calories_max,
        "calories_range": calories_range,
        "calories_p25": calories_p25,
        "calories_p75": calories_p75,
        "calories_skew": calories_skew,
        "calories_kurtosis": calories_kurtosis,

        "resting_min": resting_min,
        "resting_max": resting_max,
        "resting_range": resting_range,
        "resting_p25": resting_p25,
        "resting_p75": resting_p75,
        "resting_skew": resting_skew,
        "resting_kurtosis": resting_kurtosis,

        "finding_1": finding_1,
        "finding_2": finding_2,
        "finding_3": finding_3
    }