import pandas as pd
import ast



def generate_inverted_index(data):
    inv_idx_dict = {}
    term_count = {}
    for index, doc_text,word_count in zip(data['index'], data['text'], data['wordCount']):
        
        for word in doc_text.split():
            
            if word not in inv_idx_dict.keys():
                inv_idx_dict[word] = [index]
                term_count[word] = word_count[word]
            elif word in inv_idx_dict.keys():
                inv_idx_dict[word].append(index)
                term_count[word]+= word_count[word]

    return inv_idx_dict, term_count

newgroups_train = pd.read_csv("dataProcessed.csv",converters={'wordCount':ast.literal_eval})
res, term_count = generate_inverted_index(newgroups_train)

data = {'term': list(res.keys()),
        'indexes': res.values(),
        'termsCount': term_count.values()
}

df = pd.DataFrame(data)

df.to_csv("invertedIndexes.csv", index=False)

