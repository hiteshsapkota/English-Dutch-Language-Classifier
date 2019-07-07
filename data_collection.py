#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:41:31 2018

@author: hiteshsapkota
"""
import wikipedia

import string

import csv

langs = ['en', 'de']

write_file = open("data.csv", "w", encoding="utf-8")

writer = csv.writer(write_file)

mydata=[['sentence', 'length', 'lang']]

writer.writerows(mydata)

for lang in langs:
    
    wikipedia.set_lang(lang)
    
    titles = wikipedia.random(pages=100000)
    
    error_count = 0
    
    linesToCopy = []
    
    for title in titles:
        
        try:
            
            summary = wikipedia.summary(title)
            
            sentences = summary.split(".")
            
            for sentence in sentences:
                
                sentence = sentence.translate(str.maketrans('','',string.punctuation))
                
                sentence = sentence.translate(str.maketrans('','','1234567890'))
                
                sentence = sentence.replace('\n', ' ').replace('\r', '')
                
                sentence = ' '.join(sentence.split())
                
                lineList = sentence.split(" ")
                
                if len(lineList)>3 and len(lineList)<=50:
                    
                    sentence = " ".join(lineList[0:len(lineList)])
                    
                    mydata=[[sentence, len(lineList), lang]]
                    
                    writer.writerows(mydata)
                   
                
           
        except wikipedia.exceptions.DisambiguationError as e:
            
            print(e)
            
            error_count+=1
            
        except wikipedia.exceptions.PageError as e:
            
            print(e)
            
            error_count+=1
            
    print("Error Count for " +lang+ " = "+str(error_count))
            
            
            
            
            
        
    


