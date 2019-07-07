#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 06:39:52 2018

@author: hiteshsapkota
"""
import numpy as np
import pandas as pd
import string
from features import combine_features

predictor_names =  ['hasy', 'startD', 'repeatVowels', 'presenceij', 'presence_dutch_frequent', 'presence_english_frequent', 'avgwordlength', 'startswithAvowel', \
                    "hasdutch_diphtongs", "hasnonEnglish", "hasenglish_stopword", "hasdutch_stopword"]

"""Calculates the entropy of the given dataset
   Input paramter: target column"""
   
def entropy(target_col):

    elements,counts = np.unique(target_col,return_counts = True)
    
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    
    return entropy


"""Calculates the information gain
   Paramerers:
       1. The dataset for which we are going to calculate the information gain
       2. split_attribute_name is the split predictor name
       3. target_name is the name of the target_class default id """
       
def IG(data, split_attribute_name, target_name):
   
    total_entropy = entropy(data[target_name])
    
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)

    splitted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
 
    Information_Gain = total_entropy - splitted_Entropy
    
    return Information_Gain




"""returns the tree generated using training dataset. We use recursive algorithm to construct the tree
    Parameters: 
        1. The dataset for which we are going to construct the decision tree
        2. The subset of the data we want to split in the iteration i
        3. Features (predictors) of the dataset
        4. Depth of the tree that we want to go
        5. Name of the target_class
        6. Class of the parent node, default is None"""
        
        
def genTree(original_data, splitted_data, features, total_features, depth, target_name, parent_class = None):
    
    """Checks the termination condition"""
    
    #Checks whether the tree meets the specified depth of the tree
    if (total_features-len(features))==depth:
        
        class_idx = np.argmax(np.unique(splitted_data[target_name],return_counts=True)[1])
        
        return np.unique(splitted_data[target_name])[class_idx]
    
    #Checks if all the instance are of same category
    if len(np.unique(splitted_data[target_name])) == 1:
        return np.unique(splitted_data[target_name])[0]
    
    #Checks if we are done with all features or there is no data to split
    elif len(features) ==0 or len(splitted_data)==0:
        return parent_class
    
    
    
    
    
    else:
        
        parent_class_idx = np.argmax(np.unique(splitted_data[target_name],return_counts=True)[1])
        
        parent_class = np.unique(splitted_data[target_name])[parent_class_idx]
        
        split_feat_idx = np.argmax([IG(splitted_data, feature, target_name) for feature in features])
        
        split_feature = features[split_feat_idx]
        
        decision_tree = {split_feature:{}}
       
        #Removes the feature that is already used for the split
        
        features =[feature for feature in features if feature!=split_feature]
        
        
        for value in np.unique(splitted_data[split_feature]):
            value=value
            splitted_data_new = splitted_data.where(splitted_data[split_feature] == value).dropna()
            
            splitted_tree = genTree(original_data, splitted_data_new, features,  total_features, depth, target_name ,parent_class)
            
            decision_tree[split_feature][value] = splitted_tree
        return decision_tree








    
"""Predicts the class of the test instance
   Parameters:
       1. Dictionary attributes as a key and binary value is a value
       2. Trained model
       3. default value"""    
    
def predict(test_instance, decision_tree, default=1):
    test_keys = [k for k,v in test_instance.items()]
    decision_tree_keys = [k for k,v in decision_tree.items()]
    for test_key in test_keys:
        if test_key in decision_tree_keys:
            try:
                sub_tree = decision_tree[test_key][test_instance[test_key]]
            except:
                return default
            if isinstance(sub_tree, dict):
                return predict(test_instance,sub_tree)
            else:
                return sub_tree

"""Cross validation to get the optimal depth that maximizes the validation error
   Parameters:
       1. Dictionary attributes as a key and binary value is a value
       2. Trained model
       3. default value"""         



def CV(train_data, val_data, target_name = "Lang"):
    features = train_data.columns[:-1]
    
    K=len(features)
    
    prediction_accuracy={}
    
    models = {}
    
    for depth in range (1, K):
        
        tree = genTree(train_data, train_data,  features, len(features), depth, target_name)
        
        models[depth] = tree
        
        x_test = val_data.drop(target_name, axis=1)
        
        predicted = pd.DataFrame(columns=["predicted"]) 
        
        for i in range(0, len(x_test)):
            
            predicted.loc[i,"predicted"] = predict(x_test.iloc[i, ].to_dict(),tree,1.0)
            
        accuracy= (np.sum(predicted["predicted"] == val_data[target_name])/len(val_data))*100
        
        prediction_accuracy[depth]=accuracy
        
    opt_depth = max(prediction_accuracy.keys(), key=(lambda key: prediction_accuracy[key]))
    
    opt_tree = models[opt_depth]
    
    return [opt_depth, opt_tree, prediction_accuracy[opt_depth]]

"""Trains the decision tree and saves it
    Parameters: 1. Name of the training data
    2. Name of the validation data"""



def traintree(train_data_name, val_data_name):
    print("Training: training dataset:", train_data_name, "valid dataset:", val_data_name)
        
    train_dataset = pd.read_csv('dataset/train/'+train_data_name+".csv")
        
    val_dataset = pd.read_csv('dataset/val/'+val_data_name+".csv")
        
    train_dataset = train_dataset.drop('Length',axis=1)
        
    val_dataset = val_dataset.drop('Length',axis=1)
        
    [optimal_depth, model, accuracy] = CV(train_dataset, val_dataset)
        
    print("Optimal depth of tree is:", optimal_depth, "with accuracy:", accuracy)
        
    print("Saving Model.......")
        
    model_file_path = "models/"+"dec_train_"+train_data_name+"_val_"+val_data_name+".npy"
            
    np.save(model_file_path, model)

"""Makes prediction of the given input text
    Parameters:
    1. Sentence that you want to classify"""

def predicttree(sentence):
    """Sentence preprocessing"""
    
    sentence = sentence.translate(str.maketrans('','',string.punctuation))
    
    sentence = sentence.translate(str.maketrans('','','1234567890'))
    
    sentence = sentence.replace('\n', ' ').replace('\r', '')
    
    sentence = ' '.join(sentence.split())
    
    lineList = sentence.split(" ")
    
    sentence = " ".join(lineList[0:len(lineList)])
    
    length_sentence = len(sentence)
    
    if length_sentence<=10:
        
        model=np.load("models/"+"dec_train_10_test_10.npy").item()
    
    elif length_sentence>10 and length_sentence<=20:
        
        model=np.load("models/"+"dec_train_20_val_20.npy").item()

    else:
        model=np.load("models/"+"dec_train_20_val_50.npy").item()

    features = combine_features(sentence)

    test_instance = {}
    
    for i in range(0, len(features)):
        
        test_instance[predictor_names[i]] = features[i]

    prediction = predict(test_instance, model)

    if prediction==0:
        
        return "Dutch"

    else:
    
        return "English"

                    
    

if __name__ == "__main__":
    
    """Training and test dataset"""
    train_data_files = ['10.csv', '20.csv', '50.csv']
    
    val_data_files = ['10.csv', '20.csv', '50.csv']
    
    for train_data_file in train_data_files:
        
        for val_data_file in val_data_files:
            
            print("Working by considering: training dataset:", train_data_file, "validation dataset:", val_data_file)
            
            train_dataset = pd.read_csv('dataset/train/'+train_data_file)
            
            val_dataset = pd.read_csv('dataset/val/'+val_data_file)
            
            train_dataset = train_dataset.drop('Length',axis=1)
            
            val_dataset = val_dataset.drop('Length',axis=1)
            
            [optimal_depth, _, accuracy] = CV(train_dataset, val_dataset)
            
            print("Optimal depth of tree is:", optimal_depth, "with accuracy:", accuracy)
            
            print("______________________________________________________________________")
    
   
    
