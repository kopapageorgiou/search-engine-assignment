import glob, os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string


PATH = "./20_newsgroups"

def text_preprocess(text):
    nltk_english_stopwords = stopwords.words('english')
    tran = str.maketrans('','', string.punctuation)
    text = text.translate(tran)
    text = text.lower()
    cleaned_text = ""
    ps = PorterStemmer()
    words = word_tokenize(text)
    for word in words:
        if word not in nltk_english_stopwords:
            stemmedWord = ps.stem(word)
            cleaned_text += stemmedWord + " "
    return cleaned_text


data = {}
files = glob.glob(PATH + '/**/*', recursive=True)
#print(files)
text = ""
indexes = []
texts = []
terms = []
for file in files:
    if os.path.isfile(file):
        with open(file,"r", encoding='latin-1') as fp:
            text = fp.read()
            textReformed = text_preprocess(text)
            wordCount = {}
            for word in textReformed.split():
                if word in wordCount.keys():
                    wordCount[word] += 1
                else:
                    wordCount[word] = 1
            indexes.append(int(os.path.basename(file)))
            texts.append(textReformed)
            terms.append(wordCount)

data = {'index': indexes,
        'text': texts,
        'wordCount': terms
}
df = pd.DataFrame(data)

df.to_csv("dataProcessed.csv")
