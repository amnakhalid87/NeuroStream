import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in ["director", "cast", "country"]:
        df[col] = df[col].fillna("")

    df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
    df["duration"] = df["duration"].fillna(df["duration"].mode()[0])
    df["date_added"] = df["date_added"].fillna("Unknown")

    df["cast_trimmed"] = df["cast"].apply(trim_cast)
    df = df.drop(columns=["cast", "show_id"])

    return df


def trim_cast(cast_str: str, n: int = 3) -> str:
    if cast_str == "":
        return ""
    names = [x.strip() for x in cast_str.split(",")]
    return " ".join(names[:n])


def build_content_column(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["content"] = (
        df["title"] + " " + df["listed_in"] + " " + df["director"] + " " + df["cast_trimmed"] + " " + df["description"]
    )
    df["content"] = df["content"].str.lower().str.strip()

    return df
