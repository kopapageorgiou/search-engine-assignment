import pandas as pd
import ast
import math

def getIDF(df : int):
    N = 19997
    return math.log(N/df)

def getTFIDF(idf, tf):
    return idf * tf

inverted_indexes = pd.read_csv("ivertedIndexes.csv", converters={'indexes':ast.literal_eval})
#print(inverted_indexes)
while True:
    x = input("Enter term: ")
    if x == "exit()":
        break
    #indexes = inverted_indexes.get(x)
    indexes = inverted_indexes[inverted_indexes['term'].str.fullmatch(x, na=False)]
    df = 0
    tf = int(indexes['termsCount'])
    print(tf)
    for indexgroups in indexes['indexes']:
        df +=len(indexgroups)
    #print(type(indexes['text'].iloc[0]))
    print(f"found in {df} documents\n")
    idf = getIDF(df)
    tfIDF = getTFIDF(idf, tf)
    print(f"df = {df}")
    print(f"tf = {tf}")
    print(f"idf = {idf}")
    print(f"tf-idf = {tfIDF}")
    print(indexes)

