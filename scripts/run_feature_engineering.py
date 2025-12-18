from src.machine_learning.data_prep import prepare_ml_dataset


def main():
    df = prepare_ml_dataset()
    print(df.head())
    print("Feature engineering completed successfully")


if __name__ == "__main__":
    main()
