import os
import pandas as pd
from huggingface_hub import hf_hub_download


def load_data(path: str) -> pd.DataFrame:

    repo_file_path = hf_hub_download(
        repo_id="amnaakhalid1/Netflix_data", filename="netflix_data.csv.zip", repo_type="dataset"
    )

    return pd.read_csv(repo_file_path, encoding="latin1")
