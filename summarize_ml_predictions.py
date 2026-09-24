import pandas as pd
import numpy as np
import string
from langvec import LangVec

import sys 
import mpire
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import multiprocessing 
from mpire import WorkerPool
from pprint import pprint
import itertools
from multiprocessing import Manager


def summarize_ml_prediction(df,target_col,prediction_idx):
    LEXICON = list(string.ascii_letters)
    lv = LangVec(lexicon=LEXICON)
    df = df.drop(target_col,inplace=False,axis=1)
    vectors = df.to_numpy()
    lv.fit(vectors)
    prediction_vector = vectors[prediction_idx]
    x = lv.predict(prediction_vector)
    predicted_string = "".join(x)
    return {"predicted_string":predicted_string,"prediction_idx":prediction_idx}

def get_meaningful_summarizations(df,target_col,show_details,show_progress):

    df2 = df.copy()

    if show_details == True:
        l = len(df.columns)
        print(f"There {l} columns are present:- {list(df.columns)}, Among these the target column identified is {target_col}")
            
    N,D = df.shape    
    DIMENSIONS = D
    results = []
    for idx in range(N):
        my_dict = {
            "df":df,
            "target_col":target_col,
            "prediction_idx":idx
        }
        results.append(my_dict)
        
    num_cores = max(multiprocessing.cpu_count()//2,1)

    with WorkerPool(n_jobs=num_cores,daemon=False) as pool:
        results = pool.map(summarize_ml_prediction, results, progress_bar = show_progress)
    results2 = {}
    for my_dict in results:
        predicted_string = my_dict['predicted_string']
        prediction_idx = my_dict['prediction_idx']
        results2[predicted_string] = results2.get(predicted_string,[])+[prediction_idx]
    return results2


if __name__=="__main__":
    file_path = r"C:\Users\gprak\Downloads\Github Repos\summarize-ml-predictions\stockdata.csv"
    df = pd.read_csv(file_path)
    target_col = "Date"
    show_progress = True
    show_details = True
    results = get_meaningful_summarizations(df,target_col,show_details,show_progress)
    from pprint import pprint
    pprint(results)