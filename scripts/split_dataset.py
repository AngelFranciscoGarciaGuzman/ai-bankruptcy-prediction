from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = PROJECT_ROOT / "original-dataset" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "split-dataset"
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n_test = int(round(len(shuffled) * TEST_SIZE))
    test_df = shuffled.iloc[:n_test]
    train_df = shuffled.iloc[n_test:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_path = TRAIN_DIR / "data.csv"
    test_path = TEST_DIR / "data.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Total filas: {len(df)}")
    print(f"Train: {len(train_df)} filas -> {train_path.relative_to(PROJECT_ROOT)}")
    print(f"Test:  {len(test_df)} filas -> {test_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
