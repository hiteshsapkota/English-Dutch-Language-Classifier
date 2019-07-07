#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 07:30:42 2018

@author: hiteshsapkota
"""

import sys
from decision_tree import traintree, predicttree
from adaboost import trainada, predictada
import pandas as pd
import numpy as np

if __name__== "__main__":
    classifiers = ['ada', 'dec']
    
    if len(sys.argv)<4:
        print("Input paramter missing")
        sys.exit (1)
    else:
        classifier = sys.argv[1]
        if classifier not in classifiers:
           print("Invalid classifier type")
           sys.exit (1)
        task = sys.argv[2]
        if task=="train":
            
            try:
               
                train_data_name = sys.argv[3]
                val_data_name = sys.argv[4]
                if classifiers=='dec':
                    traintree(train_data_name, val_data_name)
                else:
                    trainada(train_data_name, val_data_name)
            
                
            except IndexError:
                print("Provide both training and validation file name")
                sys.exit (1)
            
        elif task=="predict":
            file_name = sys.argv[3]
            file_content=open(file_name)
            for text in file_content:
                if classifiers=='dec':
                    print("Provided text:", text, "is:", predicttree(text))
                else:
                    print("Provided text:", text, "is:", predictada(text))
            
        else:
            print("Please enter valid task: train or predict")
            sys.exit (1)
        
        
