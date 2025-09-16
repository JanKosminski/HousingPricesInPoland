import pandas as pd
from typing import List
import unicodedata

def validate_binary(binary_cols: List[str], dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Validate and standardize binary columns in a DataFrame.

        Converts various string representations of binary values to integers (1/0).
    Handles common typos, abbreviations, and numeric strings. Unexpected values
    are replaced with a default (0), and a warning is printed.

        Parameters
        ----------
        binary_cols : List[str]
            List of column names in the DataFrame that should contain binary values.
        dataframe : pandas.DataFrame
            The DataFrame to process.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with specified binary columns converted to integers and
            invalid values replaced with the default.
        """
    true_values = {'true', 't', 'yes', 'y', '1', 'on'}
    false_values = {'false', 'f', 'no', 'n', '0', 'off'}
    default_value = 0

    for col in binary_cols:
        dataframe[col] = dataframe[col].astype(str).str.strip().str.lower()

        dataframe[col] = dataframe[col].apply(
            lambda x: 1 if x in true_values else 0 if x in false_values else default_value
        )
        unexpected_mask = ~dataframe[col].isin([0, 1])
        if unexpected_mask.any():
            print(f"Warning: Unexpected values found in column '{col}' — replaced with default ({default_value})")

    return dataframe


def fill_with_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
        Fill missing values in numeric columns with the median of each column.

        This function identifies all numeric columns (float64 and int64 types) in
        the DataFrame and replaces any NaN values with the median value of the
        respective column.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame to process.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with missing numeric values replaced by column medians.
        """
    numerical_columns = dataframe.select_dtypes(['float64', 'int64']).columns
    dataframe[numerical_columns] = dataframe[numerical_columns].fillna(dataframe[numerical_columns].median())
    return dataframe


def fill_with_unknown(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
       Fill missing values in categorical (object) columns with the string 'Unknown'.

       This function identifies all columns of type 'object' in the DataFrame and
       replaces any NaN values with the placeholder string 'Unknown'.

       Parameters
       ----------
       dataframe : pandas.DataFrame
           The DataFrame to process.

       Returns
       -------
       pandas.DataFrame
           The DataFrame with missing categorical values replaced by 'Unknown'.
       """
    cat_columns = dataframe.select_dtypes(include='object').columns
    dataframe[cat_columns] = dataframe[cat_columns].fillna('Unknown')
    return dataframe


def remove_diacritics(text: str) -> str:
    """
    Removes diactric symbols from text
    :param text:
    :return text:
    """
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text
