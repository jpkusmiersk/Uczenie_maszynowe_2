import glob
import os
import tensorflow as tf
from termcolor import colored

##########################################
##########################################
#Function to load and preprocess the text dataset

def load_wksf_dataset(filePath):
    if filePath.endswith(".txt"):
        fileList = [filePath]
    else:
        fileList = glob.glob(os.path.join(filePath, "*.txt"))
        
    print(colored("Reading text from files: ", 'green'), fileList)
    
    dataset = tf.data.TextLineDataset(fileList)
    dataset = dataset.filter(lambda line: ~tf.strings.regex_full_match(line, ".*[~].*")) 
    dataset = dataset.filter(lambda line: ~tf.strings.regex_full_match(line, ".*[<].*"))
    dataset = dataset.map(lambda line: tf.strings.regex_replace(line, "\[[0-9]+\]", ""))
    dataset = dataset.map(lambda line: tf.strings.regex_replace(line, "\[\/\]", ""))
    
    return dataset