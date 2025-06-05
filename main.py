from src.load_data import load_high_quality_questions
from src.clean_data import clean_dataset

if __name__ == "__main__":
    data_path = "data/train.csv"
    
    #Step-1: Load Data and load HQ questions (src/load_data.py)
    hq_df = load_high_quality_questions(data_path)

    #Step-2: Clean Data
    clean_df = clean_dataset(hq_df)

    print("\n--- Cleaned Rows ---")
    print(clean_df.head())