from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import time
import ast

newgroups_train = pd.read_csv("dataProcessed.csv",converters={'wordCount':ast.literal_eval})
def text_preprocess(text):
    nltk_english_stopwords = stopwords.words('english')
    tran = str.maketrans('','', string.punctuation)
    text = text.translate(tran)
    text = text.lower()
    cleaned_text = ""
    for word in text.split():
        if word not in nltk_english_stopwords:
            cleaned_text += word + " "
    return cleaned_text

#newgroups_train["text"] = newgroups_train["te"]

#newgroups_train = fetch_20newsgroups(subset="train")
"""data_reformed = []
for data in newgroups_train.data:
    data_reformed.append(text_preprocess(data))
newgroups_train.data = data_reformed"""

def retrieve_vectors(data):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data)
    docvectors = vectors.toarray()
    print(vectors.shape)
    return vectors, docvectors

tfidf, docvectors = retrieve_vectors(newgroups_train)

def generate_inverted_index(data):
    inv_idx_dict = {}
    term_count = {}
    for index, doc_text,word_count in zip(data['index'], data['text'], data['wordCount']):
        #print(word_count)
        for word in doc_text.split():
            #print(word)
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
                term_count[word] = word_count[word]
            elif word in inv_idx_dict.keys():
                inv_idx_dict[word].append(index)
                term_count[word]+= word_count[word]
            """elif index not in inv_idx_dict[word]:
                inv_idx_dict[word].append(index)
                term_count[word] += 1"""

            #if word not in inv_idx_dict.keys():
    return inv_idx_dict, term_count

def find_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
def run_inverted_index_test(data_sizes, data, docvectors):
    data_sizes = list(range(10, 101, 10))

    inv_idx_run_times = []
    inv_idx_comparisons = []
    df = pd.DataFrame(data.data)
    for data_size in data_sizes:
        test_vectors = docvectors[:data_size]
        test_data = df.iloc[:data_size].values.tolist()
        num_of_comparisons = 0
        pairwise_similarity = []
        inv_idx_dict = generate_inverted_index(test_data)
        
        start = time.time()    

        for cur_doc_index, doc in enumerate(test_data):
            to_compare_indexes = [] 
            # find all the document indexes that have a common word with the current doc
            for word in doc[0].split():
                to_compare_indexes.extend(inv_idx_dict[word])

            # eliminate duplicates
            to_compare_indexes = list(set(to_compare_indexes))

            # calculate the similarity onlf if the id is larger than 
            # the current document id for better efficiency
            cur_doc_sims = []
            for compare_doc_index in to_compare_indexes:
                if compare_doc_index < cur_doc_index:
                    continue
                sim = find_similarity(test_vectors[cur_doc_index], test_vectors[compare_doc_index])
                num_of_comparisons += 1
                cur_doc_sims.append([compare_doc_index, sim])
            pairwise_similarity.append(cur_doc_sims)

        end = time.time()
        
        time_passed = end-start
        print("data size:", data_size)
        print("time:", time_passed)
        print("number of comparisons:", num_of_comparisons)
        print()

        inv_idx_run_times.append(time_passed)
        inv_idx_comparisons.append(num_of_comparisons)
        
    return inv_idx_run_times, inv_idx_comparisons, pairwise_similarity

def plot_results(data_sizes, run_times, comparisons):
        
    plt.figure(figsize=(9, 3))
    #plt.subplots_adjust(left=-0.2)
    
    plt.clf()
    plt.subplot(121)
    plt.plot(data_sizes, run_times)
    plt.title('Data Size vs. Runtime')
    plt.xlabel("Data Size")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)

    plt.subplot(122)
    plt.plot(data_sizes, comparisons)
    plt.title('Data Size vs. Num. of Comparisons')
    plt.xlabel("Data Size")
    plt.ylabel("Num. of Comparisons")
    plt.grid(True)
    return plt
        
data_sizes = list(range(10, 101, 10))
#ii_run_times_1, ii_comparisons_1, pairwise_similarity = run_inverted_index_test(data_sizes, newgroups_train, docvectors)
res, term_count = generate_inverted_index(newgroups_train)
#pprint(res)
"""ii_run_times_1_df = pd.DataFrame(ii_comparisons_1)
ii_run_times_1_df.to_csv('ii_run_times_1.csv')
ii_comparisons_1_df = pd.DataFrame(ii_comparisons_1)
ii_comparisons_1_df.to_csv('ii_comparisons_1.csv')
pairwise_similarity_df = pd.DataFrame(pairwise_similarity)
pairwise_similarity_df.to_csv('results.csv')
plt = plot_results(data_sizes, ii_run_times_1, ii_comparisons_1)
plt.show()"""

data = {'term': list(res.keys()),
        'indexes': res.values(),
        'termsCount': term_count.values()
}
df = pd.DataFrame(data)

df.to_csv("ivertedIndexes.csv", index=False)

#print(newgroups_train.filenames.shape)
#print(newgroups_train.data)

