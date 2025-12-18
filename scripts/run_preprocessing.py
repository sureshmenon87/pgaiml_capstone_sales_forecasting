from src.datascience.preprocessing import (
    preprocess_and_merge,
    save_master_dataset,
)


def main():
    master_df = preprocess_and_merge()
    save_master_dataset(master_df)

    print("Preprocessing and merge completed successfully")


if __name__ == "__main__":
    main()
