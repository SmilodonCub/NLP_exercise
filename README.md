# NLP_exercise

**Contents**
1. NLP_Exercise.ipynd - jupyter notebook outlining thought process & approach for the NLP Exercise
2. requirements.txt - python requirements for python scripts and notebook
3. ICD_Code_dataprep.py -  python code that reads the file and stores data in an appropriate structure
4. ICD_Code_wordembeddings.py - a python code to train a word embedding model for the ICD codes
5. find_related_terms.py - a python code to return related terms when given an ICD Code

## For use:
* download and unzip “icd10cm_order_2021.txt” to the same directory as the python code
* execute **ICD_Code_dataprep.py**. This script will preprocess the ICD Code .txt file and save a formatted pandas dataframe to the working directory as 'preprocessed.parquet'
* execute **ICD_Code_wordembeddings.py**. This script will:
  - use a Bag-of-words approach to find the TF-IDF weightings of the terms in each ICD Code description
  - use a word2vec model to generate the word embeddings of all terms
  - for each ICD Code description: use TF-IDF values to weight the word embeddings of term in the description and take the mean of all terms
* run find_related_terms.py and follow the prompts:
  - enter ICD Code (with or without decimal) 
  - enter the desired number of related terms (1-10) where relation is determined by cosine similarity of ICD Code weighted embedding and the modelled word embeddings
* rerun find_related_terms as desired. 

## For background:
* refer to the NLP_Exercise.ipynd Jupyter notebook for EDA, notes and more
* **BONUS**: the notebook has some visualizations of the ICD Code embeddings (t-SNE & UMAP)
