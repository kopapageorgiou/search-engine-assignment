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
for file in files:
    if os.path.isfile(file):
        with open(file,"r", encoding='latin-1') as fp:
            text = fp.read()
            indexes.append(int(os.path.basename(file)))
            texts.append(text_preprocess(text))

data = {'index': indexes,
        'text': texts
}
df = pd.DataFrame(data)

df.to_csv("dataProcessed.csv")
