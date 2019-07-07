# Text-classification:
Classifies given short sentence to english or dutch using: (1) Decision Tree or (2) Adaptive Boosting with a Decision Stump. The platform required for the implementation on Python3 with the required file mentioned in the requirement.txt

 ## Accuracy On a Test Set:
    1. Using Decision Tree: 98.58
    2. Using Adaboost: 96.6

 ## Files Description:
 1. data_collection.py:
     This collects the raw english and dutch sentences and stores into data.csv file
 2. data.csv:
     Contains collected data with three fields: sentence, length, lang. Sentence is a raw collected text, length is a length        of sentence, and lang is language type: en for english and de for dutch
 3. adaboost.py:
     Implementation of the adaboosting learning technique, from scratch, with a decision stump
 4. decision_tree.py:
     Imlementation of the decision tree technique, from scratch, using ID3
 5. features.py:
     Transforms sentence to features
 6. main.py:
     main program that picks classifier technique and performs one of the following:
     a. trains the given classifier with the train and test sentences with word length 10, 20, and 50 respectively
     b. predicts the given text (either english or dutch) using trained model
 7. writeup.pdf:
    Detailed explanation on: Data Collection, Preprocessing, Training, and Evaluation.
     
 ## Directories Description:
    1. dataset:
        It contains two subdirectories: train and val. Each of them have a file containing sentences of path lengths 10, 20,           and 50 respectively. 
    2. models:
       Directory that holds the trained model for decision tree and adaboost
     3. weights:
         Directory that holds the weights for the adaboost during training
         
         
## Instruction for training and evaluation:
    1. For training use following command 
     python main.py classifier_type "train" train_sentence_length val_sentence_length
       where classifier_type is "dec" for decision tree or "ada" for adaboosting
             train_sentence_length is length of the sentence you want to train with (10, 20, 50)
             val_sentence_length is length of the sentence you want to perform hyperparameter tunning with (10, 20, 50)
      For eg: to train classifier decision tree with a train sentence length 50 and val sentence length 50 use following:
               python main.py "train" "dec" "50" "50" 
               
     2. For prediction use the following command
         python main.py classifier_type "predict" file_name
          where file_name is name of the file you want to test. By default put the sentence inside the text.txt file and 
          perform prediction. For multiple sentence, use one line seperation between sentences.
          
          For eg: to make prediction using classifier adaboost with a test.txt file use following command:
                  python main.py "predict" "ada" "test.txt"  
          
Please refer to the writeup.pdf for the detail in data collection, feature extraction, and the accuracy.        
 
