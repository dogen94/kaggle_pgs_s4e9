
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def load_csv(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print(f"CSV file '{file_path}' successfully loaded.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a problem parsing the file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def encode_col(df, col, encoder=OneHotEncoder(categories="auto")):
    # Example with scikit-learn
    encoded_features = encoder.fit_transform(df[[col]])
    df[col] = encoded_features.ravel()
    return df, encoder


def embed_col(df, col, tokenizer, model, ntoken=768, fout=None):
    data = df.pop(col)
    # Make token col names
    colnames = [col + f"_{i}" for i in range(ntoken)]
    # Generate a tokens and embed
    tokens = tokenizer(data.to_list(), padding=True, truncation=True,
                       return_tensors="pt")
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    if fout:
        torch.save(embeddings, fout + ".pt")
