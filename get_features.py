#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:28:38 2018

@author: hiteshsapkota
"""

import pandas as pd

import csv

from features import *

dataset = pd.read_csv('data.csv')

write_file = open("dataset.csv", "w", encoding="utf-8")

writer = csv.writer(write_file)

mydata=[['hasy', 'startD', 'repeatVowels', 'presenceij', 'presence_dutch_frequent', 'presence_english_frequent', 'avgwordlength', 'startswithAvowel', \
         "hasdutch_diphtongs", "hasnonEnglish", "hasenglish_stopword", "hasdutch_stopword", "Lang", "Length"]]

writer.writerows(mydata)

count=0

for i in range(len(dataset)):
    
    count+=1
    
    attributes = dataset.iloc[i]
    
    sentence = attributes.sentence
    
    lang = attributes.lang
    
    length = attributes.length
    
    features = combine_features(sentence)
    
    mydata=features
    
    if lang=='en':
        
        mydata.append(1)
        
    elif lang=='de':
        
        mydata.append(0)
        
    mydata.append(length)
    
    mydata=[mydata]
    
    writer.writerows(mydata)
    
df = pd.read_csv('dataset.csv')

ds = df.sample(frac=1)

ds.to_csv('shuffled_dataset.csv')

print("Total count is:", count)
    
    


