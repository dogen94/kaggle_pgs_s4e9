# Standard lib
import regex as re

# Third Party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def _engine_regex(df):
    # Regex match engine col
    eng_regs = {
        "engine_hp": r'(\d+\.\d+)HP',
        "engine_l": r'(\d+\.\d+)L',
        "engine_v": [r'V([0-9]+)', r'([0-9]+) Cylinder'],
        "engine_e": r'(Electric)'
    }
    cols = [list(eng_regs.keys())]
    for eng_str in df["engine"].to_list():
        _cols = []
        for key,reg in eng_regs.items():
            # Try until match if list
            if isinstance(reg, list):
                for regi in reg:
                    regs = re_search(regi, eng_str)
                    if regs:
                        break
            else:
                regs = re_search(reg, eng_str)
            if regs:
                _cols.append(regs.group(1))
            else:
                _cols.append(None)
        cols.append(_cols)
    return np.array(cols)


def re_search(reg, string):
    regc = re.compile(reg)
    regs = re.search(regc, string)
    return regs


def regex_col(df, col):
    # Get regex func
    func = REGEX_FUNCS.get(col, False)
    # Call
    if func:
        cols = func(df)
    else:
        raise NotImplementedError(f"No regex func for {col}")
    # Get headers
    heads = cols[0]
    # Pop col and replace
    _ = df.pop(col)
    for i,head in enumerate(heads):
        df[head] = cols[1:,i]
    return df

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


def compute_attention_weights(embeddings):
    batch_size, seq_len, hidden_size = embeddings.shape
    # Mean embedding for the batch
    mean_embedding = embeddings.mean(dim=1)
    
    # Compute attention scores for each token
    scores = torch.bmm(embeddings, mean_embedding.unsqueeze(2)).squeeze(2)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=1)
    
    return attention_weights

def apply_attention_weights(embeddings, attention_weights):
    # Compute weighted mean of token embeddings
    weighted_embeddings = torch.bmm(attention_weights.unsqueeze(1), embeddings).squeeze(1)
    return weighted_embeddings


REGEX_FUNCS = {
    "engine": _engine_regex,
}