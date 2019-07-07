#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:32:24 2018

@author: hiteshsapkota
"""

import re

from nltk.corpus import stopwords 

vowels=['A', 'E', 'I', 'O', 'U']
dutch_diphtongs = ['ae', 'ei', 'au', 'ai', 'eu', 'ie', 'oe', 'ou'\
                   'ui', 'aai', 'oe', 'ooi', 'eeu', 'ieu']

frequent_dutch_words =['ik', 'je', 'het', 'de', 'is', 'dat', 'een', 'niet', 'en', 'wat']

frequent_english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']

stop_words_english = set(stopwords.words('english')) 

stop_words_dutch = set(stopwords.words('dutch'))




        
"""Checks whether given sentence contains y
Paramters: sentence in the form of tokens"""

def hasy(tokens):
    
    for token in tokens:
        
        if 'y' in token.lower():
            
            return 1
        
    return 0


"""Checks whether any word starts with D
Parameters: sentence in the form of tokens"""
    
def startD(tokens):
    
    for token in tokens:
        
        if token[0].upper()=='D':
            
           return 1
       
    return 0


"""Checks whether sentence has word with repeated vowel
Parameters: sentence in the form of tokens"""  
      
def repeatVowels(tokens):
    
    for token in tokens:
        
        token = token.upper()
        
        if 'AA' in token or 'EE' in token or 'II' in token or 'OO' in token or 'UU' in token:
            
            return 1
    return 0

"""Checks whether any word has consecutive i and j characters
Paramters: sentence in the form of tokens"""
    
def presenceij(tokens):
    
    for token in tokens:
        
        token = token.upper()
        
        if 'IJ' in token:
            
            return 1
        
    return 0


"""Checks whether the sentence has any Dutch frequent word from the list of top 10 words
ParametersL sentence in the form of tokens"""

def presence_dutch_frequent(tokens):
    
    for token in tokens:
        
        token=token.lower()
        
        if token in frequent_dutch_words:
            
            return 1
        
    return 0


"""Checks whether the sentence has any English frequent word from the list of top 10 words
Parameters: sentence in the form of tokens"""
   
def presence_english_frequent(tokens):
    
    for token in tokens:
        
        token=token.lower()
        
        if token in frequent_english_words:
            
            return 1
        
    return 0

""""Finds if average world length is greater then 8.5
 Parameters: sentence in the from of tokens"""
 
def avgwordlength(tokens):
    
    avg_len_word=0
    
    for token in tokens:
        
        avg_len_word+=len(token)
        
    avg_len_word= avg_len_word/len(tokens)
    
    if avg_len_word>8.5:
        
        return 1
    
    else:
        
       return 0
        
   
    
"""Finds if any word starts with a vowel
Parameters: sentence in the form of tokens"""

def startswithAvowel(tokens):
    
    if tokens[0][0].upper() in vowels:
        
        return 1
    
    else:
        
        return 0
    

"""Finds whether any word are from Dutch Diphtongs
Parameters: sentence in the form of tokens"""

def presencedutch_diphtongs(tokens):
    
    for token in tokens:
        
        token=token.lower()
        
        for diph in dutch_diphtongs:
            
            if diph in token:
                
                return 1
    return 0
                
            
    
""""Checks whether any word contains non english characters
Parameters: sentence in the form of tokens"""

def presencenonEnglish(tokens):
    
    for token in tokens:
        
        if not re.match("^[A-Za-z0-9_-]*$", token):
            
            return 1
        
    return 0
            

""""Checks whether any word contains english stopword
Parameters: sentence in the form of tokens"""

def hasenglish_stopword(tokens):
    
    for token in tokens:
        
        if token in stop_words_english:
            
            return 1
        
    return 0


""""Checks whether any word contains dutch stopword
Parameters: sentence in the form of tokens"""

def hasdutch_stopword(tokens):
    
    for token in tokens:
        
        if token in stop_words_dutch:
            
            return 1
    return 0
            
""""Extracts the features using different functions and combines them
 Parameters: sentence in the form of tokens"""           
def combine_features(sentence):
    
    tokens = sentence.split(' ')
    
    features=[]
    
    features.append(hasy(tokens))
    
    features.append(startD(tokens))
    
    features.append(repeatVowels(tokens))
    
    features.append(presenceij(tokens))
    
    features.append(presence_dutch_frequent(tokens))
    
    features.append(presence_english_frequent(tokens))
    
    features.append(avgwordlength(tokens))
    
    features.append(startswithAvowel(tokens))
    
    features.append(presencedutch_diphtongs(tokens))
    
    features.append(presencenonEnglish(tokens))
    
    features.append(hasenglish_stopword(tokens))
    
    features.append(hasdutch_stopword(tokens))
    
    return features
           

    
