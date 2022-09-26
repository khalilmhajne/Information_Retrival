# -*- coding: utf-8 -*-
#import threading
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
import numpy as np
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from google.cloud import storage
import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

from inverted_index_gcp import *
from inverted_index_gcp import read_posting_list
from nltk.stem import LancasterStemmer
lancaster=LancasterStemmer()



####################################### DONE IMPORTS #######################
## initialize all necessary dicts :- 

bodyInvertedIndex=InvertedIndex.read_index("posting_lists/206416380/postings_gcp","index") 
stemIndex=InvertedIndex.read_index("posting_lists/stemmingindex206416380/postings_gcp","index") 
# titleInvertedIndex=InvertedIndex.read_index("posting_lists/titleindex206416380/postings_gcp","index")  ## Title Inverted Index 
# anchorInvertedIndex=InvertedIndex.read_index("posting_lists/anchorinvertedtext206416380/postings_gcp","index")  ## Anchor inverted Index 
stemInvertedIndex=InvertedIndex.read_index("posting_lists/stemmingindex206416380/postings_gcp","index")
###########################################################################################################
a_file = open("posting_lists/206416380/dictionaries/titeIdDict.pkl", "rb")
titleDictId = pickle.load(a_file)
a_file = open("posting_lists/206416380/dictionaries/DL.pkl", "rb")
DL = pickle.load(a_file)
a_file = open("posting_lists/206416380/dictionaries/pageRank.pkl", "rb")
pageRank = pickle.load(a_file)
a_file = open("posting_lists/206416380/dictionaries/pageView.pkl", "rb")
pageView = pickle.load(a_file)
a_file = open("posting_lists/206416380/dictionaries/docWI.pkl", "rb")
docWI = pickle.load(a_file)

a_file = open("posting_lists/206416380/dictionaries/stemWI.pkl", "rb") ## TODO 
stemWI = pickle.load(a_file)
###########################################################################################################
bodyPostingPath="/home/ehsankittany/posting_lists/206416380/postings_gcp/"
titlePostingPath="/home/ehsankittany/posting_lists/titleindex206416380/postings_gcp/"
anchorPostingPath="/home/ehsankittany/posting_lists/anchorinvertedtext206416380/postings_gcp/"
stemPostingsPath="/home/ehsankittany/posting_lists/stemmingindex206416380/postings_gcp/"
############################################################################################################


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text,stem=False):
    
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    newTok=[]
    for i in tokens:
        if i in all_stopwords:
            continue
        else:
            if(stem):
              
              nword=lancaster.stem(i)
              
              newTok.append(nword)
             
            else:
              newTok.append(i)
    
    return newTok



def binarySearch(query,inIndex,postingListPath): ## Search By title 
  #query="python Foster "
  
  tokensQuery=tokenize(query)  ## Tokenize Query and remove Stop words 
  postingLists=[]

  #os.chdir("/content/posting_lists/titleindex206416380/postings_gcp/")  
  os.chdir(postingListPath)    
  for word in tokensQuery: ## Get each posting list of word in query 
    try: 
      inIndex.df[word]
      postingLists.append(read_posting_list(inIndex,word))
    except:
      postingLists.append([])

  dictTemp={}
  counter=1
  for ps in postingLists:   ## Count word repetition in titles  
    for w in ps:
      if w[0] not in dictTemp.keys():
        dictTemp[w[0]]=counter
      else:
        dictTemp[w[0]]+=counter
      
  try:
      dictTemp.pop(0)
  except:
      pass
  
  itemsLst=list(dictTemp.items()) 
  itemsLst.sort(key=lambda x : x[1],reverse=True) ## order By repetition 
  #result=itemsLst[:k] ## Return top 100 
  
  out=[(x[0]," ".join(titleDictId[x[0]])) for x in itemsLst]  ## Add Title to tuple 
  os.chdir("/home/ehsankittany")
  return out



def searchBody(query):
  
  query=tokenize(query)
  os.chdir(bodyPostingPath)

  postingLists=[]
  for word in query:
      try:
          postingLists+=read_posting_list(bodyInvertedIndex,word)
      except:
         pass

  queryLen=(len(query))**0.5
  
  scoresList={}
  

  for id,tf in postingLists:

    denominator=docWI[id]**0.5 *queryLen
    numenator=tf/ DL[id]
    score=numenator/denominator
    
    try:
      scoresList[id]+=score
    except :
      scoresList[id]=score

  
  scoresList=list(scoresList.items())

  scoresList=sorted(scoresList,key=lambda x:x[1],reverse=True)
  scoresList=scoresList[:100]
  output=[(id," ".join(titleDictId[id])) for id,cosine in scoresList]


  os.chdir("/home/ehsankittany")
  return output



################################################################################

commonWords=["make","kill","come","made","best","big","small","tiny","worst","works"]
def getCosineSim(query,inIndex,WIdic,postingsPath,stem=False):

  query=tokenize(query,stem)
  os.chdir(postingsPath)

  postingLists=[]
  for word in query:
    postingLists.append((word,read_posting_list(inIndex,word)))

  queryLen=(len(query))**0.5  

  scoresList={}
  flag =False
  for word,ps in postingLists:
    if word in commonWords:
          flag=True
    for id,tf in ps:
    
      denominator=WIdic[id]**0.5 *queryLen
      numenator=tf/ DL[id]
      score=numenator/denominator

      if(flag):
        score=score*0.3

      try:
        

        scoresList[id]+=score
      except :
        scoresList[id]=score

    flag=False

  

  scoresList=list(scoresList.items())

  os.chdir("/home/ehsankittany")

  scoresList=sorted(scoresList,key=lambda x:x[1],reverse=True)
  scoresList=scoresList[:]
  return scoresList
  








def searchStemBody(query,titleInvertedIndex,anchorIndex,bodyWeight=5,anchWieght=0.2,titleWeight=0.5):
  
  stemD=dict(getCosineSim(query,stemIndex,stemWI,stemPostingsPath,True))
  bodycosineDt=dict(getCosineSim(query,bodyInvertedIndex,docWI,bodyPostingPath))
  titledict=dict(binarySearch(query,titleInvertedIndex,titlePostingPath))
  anchorDict=dict(binarySearch(query,anchorIndex,anchorPostingPath))

  idSet=set(list(bodycosineDt.keys()))


  # bodyWeight=0.2
  stemWeight=1.5
  # titleWeight=0.6
  
  output={}
  for id in idSet:
    score=0
    # if stemD.get(id):
    #   score+=(stemD.get(id)*stemWieght)
    if stemD.get(id):
      score+=(stemD.get(id)*stemWeight)
    if bodycosineDt.get(id):
      score+=(bodycosineDt.get(id)*bodyWeight)

    if titledict.get(id):
      score+=(1*titleWeight)
    if anchorDict.get(id):
        score+=(1*anchWieght)
    
    output[id]=score
  

  output=list(output.items())
  output=sorted(output,key=lambda x:x[1],reverse=True)
  output=[(id," ".join(titleDictId[id])) for id,_ in output[:100]]
    

  
  return output

















