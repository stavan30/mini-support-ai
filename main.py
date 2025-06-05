from src.load_data import load_high_quality_questions

if __name__ == "__main__":
    data_path = "data/train.csv"
    hq_df = load_high_quality_questions(data_path)

    print("\n--- Sample Rows ---")
    print(hq_df.head())