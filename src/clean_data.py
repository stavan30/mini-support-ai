import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_html(raw_html):
    """Remove HTML tags and decode HTML entities."""
    if pd.isna(raw_html):
        return ""
    return BeautifulSoup(raw_html, "html.parser").get_text()

def extract_tags(tag_string):
    """Convert <python><docker> into ['python', 'docker']"""
    return re.findall(r'<(.*?)>', tag_string)

def clean_dataset(df):
    df["Title"] = df["Title"].fillna("").str.strip().str.lower()
    df["Body"] = df["Body"].apply(clean_html).str.strip().str.lower()
    df["Tags"] = df["Tags"].fillna("").apply(extract_tags)
    return df
