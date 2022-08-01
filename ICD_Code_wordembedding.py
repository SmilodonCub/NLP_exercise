# ICD_Code_wordembedding
# a python code to train a word embedding model for the ICD codes using the code descriptions

# import requirements
import pandas as pd
import numpy as np
import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# embedding functions
def find_tfidf_weights(text_data_list):
    """
    use a bag-of-words approach to find tf-idf weights for each document
    """
    # use a dictionary to build the bag-of-words representation of the text
    dct = Dictionary(text_data_list)
    ICD_text_bow = [dct.doc2bow(line) for line in text_data_list]
    # use gensims TfidfModel to find the tfidf values for the terms in each doc
    model_tfidf = TfidfModel(ICD_text_bow)
    # the results come as a tuple (term, tfidf_val)
    # reformat as a list of tfidf_val for each doc
    tfidf_weights = [
        [w[1] for w in model_tfidf[ICD_text_bow[index]]]
        for index in range(0, len(ICD_text_bow))
    ]
    return tfidf_weights


def weighted_ICD(terms, tfidf_weights, wordembed):
    """
    weighted_ICD finds the weighted embedding for an ICD Code long description
    given a list of terms (list of string) of len t
    given a list of tfidf_weights (list of float) of len t
    given a word2vec_model which gives embeddings of len v
    returns a weighted mean vector of len v
    """
    # an array of embeddings for each term in terms
    terms_embeddings = wordembed[terms]
    # tfidf_weights as an array
    tfidf_weights = np.array(tfidf_weights)

    # finding the weighted mean to give an embedded representation of terms
    broad_weight = np.broadcast_to(tfidf_weights, terms_embeddings.T.shape).T
    weighted_vec = broad_weight * terms_embeddings
    weighted_vec = weighted_vec.mean(axis=0)

    return weighted_vec


def all_weighted_ICD(df, wordembed):
    """
    find weighted_ICD embeddings for each row in df
    """
    weighted_embeddings = []
    for index, row in df.iterrows():
        processed_terms = row["Processed_Long_Description"]
        # remove duplicate terms (keep first)
        processed_terms = [
            i for n, i in enumerate(processed_terms) if i not in processed_terms[:n]
        ]
        weights = row["tfidf_weights"]
        try:
            res = weighted_ICD(processed_terms, weights, wordembed)
        except:
            print(index)
        weighted_embeddings.append(res)
    return weighted_embeddings


# main
def main():

    df = pd.read_parquet("preprocessed.parquet")

    # training the word2vec skipgram word embedding model prepare the text data as a list
    ICD_text_data = [list(t) for t in df["Processed_Long_Description"]]

    # train model
    print(
        "TRAINING WORD2VEC: \nusing skip-gram to generate word embeddings trained with the ICD Code Long Descriptions\n\nthis may take a minute or two..."
    )
    w2v_model = gensim.models.Word2Vec(ICD_text_data, min_count=1, workers=3, sg=1)
    wordembeddings = w2v_model.wv

    # find the tf-idf values for terms in each text document (idc code description) add the df
    print(
        "\nFINDING TF-IDF: \ntf-idf will be used to weight the word embeddings of the terms in each ICD Code description"
    )
    df["tfidf_weights"] = find_tfidf_weights(ICD_text_data)

    # find the mean tfidf weighted embedding of the terms for each ICD Code description
    print(
        "\nGENERATING ICD CODE EMBEDDING: \ntaken as the mean of the tf-idf weighted terms in each ICD Code description"
    )
    weighted_embeddings = all_weighted_ICD(df, wordembeddings)
    df["weighted_embeddings"] = weighted_embeddings

    # save the w2v word embeddings and df for use by find_related_terms.py
    df.drop(columns=["tfidf_weights"])
    filename = "processed.parquet"
    df.to_parquet(filename)
    print("\nProcessed data saved to working directory as 'processed.parquet'")
    wordembeddings.save("wordembeddings")
    print("Word vectors (w2v_model.wv) saved to working directory as 'wordembeddings'")


if __name__ == "__main__":
    main()
