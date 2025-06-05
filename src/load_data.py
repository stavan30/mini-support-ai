import pandas as pd

def load_high_quality_questions(csv_path):
    """
    Load the dataset and return only high-quality questions.
    """
    df = pd.read_csv(csv_path)
    print("Original Shape:", df.shape)

    #Filter only HQ(High Quality questions - Tag from Dataset)
    hq_df = df[df["Y"] == "HQ"].copy()
    print("High-quality only:", hq_df.shape)

    # Return just useful columns
    return hq_df[["Title", "Body", "Tags"]]