from pathlib import Path
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "students.csv"
MODEL_FILE = HERE / "student_model.joblib"
FEATURES = ["attendance", "previous_marks", "study_hours"]

def train_model():
    data = pd.read_csv(DATA_FILE)
    required = FEATURES + ["final_marks"]
    data = data.dropna(subset=required)
    if len(data) < 10:
        print("At least 10 complete rows are required.")
        return False
    X, y = data[FEATURES], data["final_marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f} marks")
    print(f"R²: {r2_score(y_test, pred):.3f}")
    joblib.dump(model, MODEL_FILE)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, pred)
    plt.xlabel("Actual final marks")
    plt.ylabel("Predicted final marks")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(HERE / "prediction_chart.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(FEATURES, model.feature_importances_)
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(HERE / "feature_importance.png")
    plt.close()
    print("Saved model and two charts.")
    return True

def predict(attendance, previous_marks, study_hours):
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Train the model first.")
    model = joblib.load(MODEL_FILE)
    sample = pd.DataFrame([{
        "attendance": attendance,
        "previous_marks": previous_marks,
        "study_hours": study_hours
    }])
    return float(model.predict(sample)[0])
