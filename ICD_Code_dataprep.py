# ICD_Code_dataprep
# a python code that reads the file and stores data in an appropriate structure

# import requirements
import pandas as pd
import os
import string

# preprocessing functions
def clean_ICDCode(df):
    """
    clean whitespace from ICD_Code feature
    """
    df["ICD_Code"] = df["ICD_Code"].str.strip()
    return df


def add_initial(df):
    """
    add a new feature for the first letter of the ICD_Codes
    """
    first_letters = [c[0] for c in df["ICD_Code"]]
    df["Initial"] = first_letters
    return df


def preprocess_longdescription(df):
    """
    preprocess the long descriptions
    add as a new feature to df
    """
    # strip leading and lagging whitespace, lowercase text and remove punctuation
    ICD_text_data = [
        sent.strip().lower().translate(str.maketrans("", "", string.punctuation))
        for sent in df["Long_Description"]
    ]
    # split into words on whitespace
    ICD_text_data = [sent.split(" ") for sent in ICD_text_data]
    # remove empty strings
    ICD_text_data = [[s for s in lst if len(s) >= 1] for lst in ICD_text_data]
    # add as a new feature to df
    df["Processed_Long_Description"] = ICD_text_data
    return df


def preprocess_df(df):
    """
    perform preprocessing steps on df
    1. clean ICD_Code
    2. add a field for the first initial of the ICD_Code
    (used for visualization of the embeddings)
    3. preprocess the 'Long_Description' text data
    """
    df = clean_ICDCode(df)
    df = add_initial(df)
    df = preprocess_longdescription(df)
    return df


def saveselect_asparquet(df, filename):
    """
    save a subset of the preprocessed df as parquet file
    restrict to features used for word embedding
    """
    keep_cols = ["ICD_Code", "Initial", "Processed_Long_Description"]
    df_sub = df[keep_cols]
    df_sub.to_parquet(filename)


# main
def main():
    # execute shell command to reformat icd10cm_order_2021.txt
    os.system(
        "sed 's/./;/6; s/./;/14; s/./;/16; s/./;/77' icd10cm_order_2021.txt > icd10cm_order_2021_cdelim.txt"
    )

    # load data as a pandas dataframe
    data_file = "icd10cm_order_2021_cdelim.txt"
    # column names derived from the ICD10OrderFile.pdf documentation
    col_names = [
        "Order_Number",
        "ICD_Code",
        "HIPAA_bool",
        "Short_Description",
        "Long_Description",
    ]
    df = pd.read_csv(
        data_file,
        sep=";",
        header=None,
        index_col=False,
        names=col_names,
        dtype={
            "Order_Number": int,
            "ICD_Code": str,
            "HIPAA_bool": int,
            "Short_Description": str,
            "Long_Description": str,
        },
    )

    df = preprocess_df(df)
    filename = "preprocessed.parquet"
    saveselect_asparquet(df, filename)
    print("Preprocessed data saved to working directory as 'preprocessed.parquet'")


if __name__ == "__main__":
    main()
