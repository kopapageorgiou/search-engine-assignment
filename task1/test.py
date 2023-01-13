import glob, os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string


PATH = "./20_newsgroups"

files = glob.glob(PATH + '/**/*', recursive=True)
count = 0
for file in files:
    if os.path.isfile(file):
        count+=1

print(count)
