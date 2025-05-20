import pandas as pd


def explore_csv(path: str):
    # Load the CSV
    df = pd.read_csv(path)

    # Display basic info
    print("\n🔍 Basic Info:")
    print(df.info())

    # Show the first 5 rows
    print("\n📌 Sample Rows:")
    print(df.head())

    # Show column names
    print("\n🧾 Columns:")
    print(df.columns.tolist())

    # Check for missing values
    print("\n❓ Null Values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    explore_csv("data/Indian_Constitution.csv")
