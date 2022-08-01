# find similar words/terms for a given ICD Code

# dependencies
import pandas as pd
from gensim.models import KeyedVectors
import re


def related_terms(embedding, wordvectors, num_terms):
    """
    related_terms finds 'num_terms' similar terms for an ICD Code
    where similar is given by cosine similarity
    given embedding (array) of len v
    given a word2vec model which gives embeddings of len v
    given a num_terms (int)
    return a list of tuples of len num_terms
    each tuple holds a term (string) and cosine similarity value (float)
    """
    term_tups = wordvectors.most_similar(embedding)[:num_terms]
    return [t[0] for t in term_tups]


def main():
    # load data and word2vec wordvectors
    df = pd.read_parquet("processed.parquet")
    wv = KeyedVectors.load("wordembeddings", mmap="r")

    # get user input for the ICD Code & number of related terms to return
    while True:
        try:
            ICD_code = input("Enter a valid ICD Code: ")
            # remove non alphanumeric chars (if decimal place)
            ICD_code = re.sub(r"[^\w\s]", "", ICD_code)
            embedding = df[df["ICD_Code"] == ICD_code]["weighted_embeddings"].item()
            if len(embedding) < 1:
                raise ValueError("Please try again with a valid ICD Code")
            break
        except (ValueError, IndexError):
            print("Could not complete request\nPlease try again with a valid ICD Code")

    while True:
        try:
            num_terms = int(
                input("Enter number of relevant terms (numeric integer 1-10): ")
            )
            break
        except ValueError:
            print("Please enter number as a numeric integer. For example: 2, 5 or 10")

    # generate num_terms of related words determined by cosine similarity
    # print the results
    res = related_terms(embedding, wv, num_terms)
    print(res)


if __name__ == "__main__":
    main()
