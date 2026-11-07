from pathlib import Path
import pandas as pd
from model import train_model, predict

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "students.csv"

def add_record():
    a = float(input("Attendance (0-100): "))
    m = float(input("Previous marks (0-100): "))
    h = float(input("Study hours per day (0-24): "))
    final = float(input("Known final marks (0-100): "))
    if not (0 <= a <= 100 and 0 <= m <= 100 and 0 <= h <= 24 and 0 <= final <= 100):
        raise ValueError("Values are outside allowed ranges.")
    data = pd.read_csv(DATA_FILE)
    data.loc[len(data)] = [a, m, h, final]
    data.to_csv(DATA_FILE, index=False)
    print("Record saved. Retrain to include it in the model.")

def main():
    while True:
        print("\nSTUDENT PERFORMANCE PREDICTION")
        print("1. Add record  2. View records  3. Train/evaluate")
        print("4. Predict marks  5. Exit")
        c = input("Choose: ").strip()
        try:
            if c == "1":
                add_record()
            elif c == "2":
                print(pd.read_csv(DATA_FILE).to_string(index=False))
            elif c == "3":
                train_model()
            elif c == "4":
                a = float(input("Attendance (0-100): "))
                m = float(input("Previous marks (0-100): "))
                h = float(input("Study hours per day (0-24): "))
                if not (0 <= a <= 100 and 0 <= m <= 100 and 0 <= h <= 24):
                    raise ValueError("Values are outside allowed ranges.")
                print(f"Predicted marks: {predict(a, m, h):.2f}/100")
                print("Use this as a learning estimate, not a final academic decision.")
            elif c == "5":
                break
            else:
                print("Choose 1–5.")
        except (ValueError, OSError, FileNotFoundError) as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
