#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:50:49 2018

@author: hiteshsapkota
"""
import numpy as np

import pandas as pd

import string

from features import combine_features

predictor_names =  ['hasy', 'startD', 'repeatVowels', 'presenceij', 'presence_dutch_frequent', 'presence_english_frequent', 'avgwordlength', 'startswithAvowel', \
                    "hasdutch_diphtongs", "hasnonEnglish", "hasenglish_stopword", "hasdutch_stopword"]
"""Calculates the entropy
Parameters: 1. Target column you want to compute the entropy
            2. Weight of the examples"""
def entropy(target_col, w):

    [elements,_] = np.unique(target_col,return_counts = True)
    
    counts = [0]*len(elements)
    
    for k in range(len(elements)):
        
        counts[k]=np.sum(w[i] for i in range(0, len(target_col)) if target_col[i]==elements[k])
        
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    
    return entropy

"""Computes the information gain
Parameters: 1. Data for which you want to find the information gain
            2. List of the features
            3. Weight of the examples
            4. Target name"""
            
def IG(data, feature, w, target_name):
    
    data['weight'] = w
    
    total_entropy = entropy(data[target_name], w)
    
    [vals,_]= np.unique(data[feature],return_counts=True)
    
    counts=[0]*len(vals)
    
    splitted_weight=[0]*len(vals)
    
    for k in range(len(vals)):
        
        counts[k]=np.sum(w[i] for i in range(0, len(data[feature])) if data[feature][i]==vals[k])
        
        splitted_weight[k]=data.where(data[feature]==vals[k]).dropna()['weight'].reset_index(drop=True)
    
        
    splitted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[feature]==vals[i]).dropna()[target_name].reset_index(drop=True), splitted_weight[i]) for i in range(len(vals))])
    
    Information_Gain = total_entropy - splitted_Entropy
    
    return Information_Gain

"""Decision tree with the depth of 1
Parameters: 1. Dataset 
            2. Weight of the examples
            3. Features
            4. Target name"""

def Decision_Stump(dataset, w, N, features, target_name):
    
    split_feat_idx = np.argmax([IG(dataset, feature, w, target_name) for feature in features])
    
    split_feature = features[split_feat_idx]
    
    decision_tree = {split_feature:{}}
    
    [vals,_]= np.unique(dataset[split_feature],return_counts=True)
    
    splitted_weight=[0]*len(vals)
    
    for k in range(len(vals)):
        
        splitted_weight=dataset.where(dataset[split_feature]==vals[k]).dropna()['weight'].reset_index(drop=True)
        
        target_col=dataset.where(dataset[split_feature]==vals[k]).dropna()[target_name].reset_index(drop=True)
        
        [elements,_] = np.unique(target_col,return_counts = True)
        
        counts = [0]*len(elements)
        
        for l in range(len(elements)):
            
            counts[l]=np.sum(splitted_weight[i] for i in range(0, len(target_col)) if target_col[i]==elements[l])
            
        max_index = np.argmax(counts)
        
        decision_tree[split_feature][vals[k]]=elements[max_index]
        
    return decision_tree
        
    
    
        
        
"""Predicts the class of the test instance based on the trained stump decision_tree
   Parameters: 1. Trained decision stump
               2. Test instance"""
               
def predict(decision_tree, test_instance, default=1):
    
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


"""Adaboost algorithm
   Parameters: 1. Training dataset
               2. Number of decision stumps
               3. Features list
               4. Target name"""
               
               
def ADABOOST(dataset, K, features, target_name):
    
    """Here h[k] holds tree of the decision stump
            z[k] holds the weight of those stumps"""
    N=len(dataset)
    
    w = np.empty(N)
    
    w.fill(1/N)
    
    h=[0]*K
    
    z=np.empty(K)
    
    train_instances = dataset.drop(target_name, axis=1)
    
    for k in range(0, K):
        
        h[k] = Decision_Stump(dataset, w, N , features, target_name)
        
        error=0
        
        for j in range(0, N):
            
            if predict(h[k], train_instances.iloc[j, ].to_dict())!=dataset[target_name][j]:
                
                error = error+w[j]
                
        for j in range(0, N):
            
            if predict(h[k], train_instances.iloc[j, ].to_dict())==dataset[target_name][j]:
                
                w[j]=w[j]*error/(1-error)
                
        w = [w[i]/np.sum(w) for i in range(0, len(w))]
        
        z[k] = np.log((1-error)/error)
        
    return [h, z]

"""Experiments with the different number of the deicision stumps and comes with the optimal one
Parameters: 1. Training data
            2. Validation data
            3. Target name default is Lang"""

def CV(train_data, val_data, target_name="Lang"):
    features = train_data.columns[:-1]
    
    K=len(features)
    
    prediction_accuracy={}
    
    features = train_data.columns[:-1]
    
    hs={}
    
    zs={}
    
    for tree in range (1, K):
        
        [h, z] = ADABOOST(train_data, tree,  features, target_name)
        
        hs[tree]=h
        
        zs[tree]=z
        
        x_val = val_data.drop('Lang', axis=1)
        
        predicted = pd.DataFrame(columns=["predicted"])
        
        for i in range(0, len(x_val)):
            
            tree_wise_prediction=[0]*tree
            
            for j in range(0, tree):
                
                tree_wise_prediction[j] = predict(h[j], x_val.iloc[i, ].to_dict())
                
            vals = np.unique(tree_wise_prediction)
            
            weight =[0]*len(vals)
            
            for k in range(len(vals)):
                
                weight[k] = np.sum(z[j] for j in range(0, tree) if tree_wise_prediction[j]==vals[k])
                
            max_index = np.argmax(weight)
            
            predicted.loc[i,"predicted"] = vals[max_index]
            
        accuracy= (np.sum(predicted["predicted"] == val_data["Lang"])/len(val_data))*100
        
        prediction_accuracy[tree]=accuracy
        
    opt_tree = max(prediction_accuracy.keys(), key=(lambda key: prediction_accuracy[key]))
    
    opt_h = hs[opt_tree]
    
    opt_w=zs[opt_tree]
    
    return [opt_tree, opt_h, opt_w, prediction_accuracy[opt_tree]]
                

def trainada(train_data_name, val_data_name):
    
    print("Training: training dataset:", train_data_name, "valid dataset:", val_data_name)
    
    train_dataset = pd.read_csv('dataset/train/'+train_data_name+".csv")
        
    val_dataset = pd.read_csv('dataset/val/'+val_data_name+".csv")
        
    train_dataset = train_dataset.drop('Length',axis=1)
        
    val_dataset = val_dataset.drop('Length',axis=1)
        
    [optimal_depth, model, weight, accuracy] = CV(train_dataset, val_dataset)
        
    print("Optimal depth of tree is:", optimal_depth, "with accuracy:", accuracy)
        
    print("Saving Model.......")
        
    model_file_path = "models/"+"ada_train_"+train_data_name+"_val_"+val_data_name+".npy"
            
    np.save(model_file_path, model)
            
    weight_file_path = "weights/"+"ada_train_"+train_data_name+"_val_"+val_data_name+".npy"
                
    np.save(weight_file_path, weight)

"""Makes the prediction of the given sentence
    Parameters: Sentence you want to classify"""

def predictada(sentence):
    """"Preprocessing"""
    
    sentence = sentence.translate(str.maketrans('','',string.punctuation))
    
    sentence = sentence.translate(str.maketrans('','','1234567890'))
    
    sentence = sentence.replace('\n', ' ').replace('\r', '')
    
    sentence = ' '.join(sentence.split())
    
    lineList = sentence.split(" ")
    
    sentence = " ".join(lineList[0:len(lineList)])
    
    length_sentence = len(sentence)
    
    """Loads the saved decision stump and the corresponding weight of those stumps"""
    
    if length_sentence<=10:
        
        model=np.load("models/"+"ada_train_10_val_10.npy")
        
        z=np.load("weights/"+"ada_train_10_val_10.npy")
    
    elif length_sentence>10 and length_sentence<=20:
        
        model=np.load("models/"+"ada_train_20_val_20.npy")
        
        z=np.load("weights/"+"ada_train_20_val_20.npy")
    
    else:
        
        model=np.load("models/"+"ada_train_50_val_50.npy")
        
        z=np.load("weights/"+"ada_train_50_val_50.npy")

    features = combine_features(sentence)

    test_instance = {}
    
    for i in range(0, len(features)):
        
        test_instance[predictor_names[i]] = features[i]

    stump_wise_prediction=[0]*len(model)

    for j in range(0, len(model)):
        
        stump_wise_prediction[j] = predict(model[j], test_instance)

    vals = np.unique(stump_wise_prediction)

    weight =[0]*len(vals)
    
    for k in range(len(vals)):
        
        weight[k] = np.sum(z[j] for j in range(0, len(model)) if stump_wise_prediction[j]==vals[k])

    max_index = np.argmax(weight)

    prediction = vals[max_index]
    
    
    if prediction==0:
        
        return "Dutch"

    else:
    
        return "English"


            
                
 
       
    
if __name__ == "__main__":
    
    train_data_files = ['10.csv', '20.csv', '50.csv']
    
    test_data_files = ['10.csv', '20.csv', '50.csv']
    
    for train_data_file in train_data_files:
        
        for test_data_file in test_data_files:
            
            print("Working by considering: training dataset:", train_data_file, "test dataset:", test_data_file)
            
            train_dataset = pd.read_csv('dataset/train/'+train_data_file)
            
            test_dataset = pd.read_csv('dataset/val/'+test_data_file)
            
            train_dataset = train_dataset.drop('Length',axis=1)
            
            test_dataset = test_dataset.drop('Length',axis=1)
            
            [optimal_depth, _, _, accuracy] = CV(train_dataset, test_dataset)
            
            print("Optimal depth of tree is:", optimal_depth, "with accuracy:", accuracy)
            
            print("______________________________________________________________________")
    
