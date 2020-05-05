
import numpy as np
import pandas as pd

def mem_usage(data):
    if isinstance(data,pd.DataFrame):
        usage_b = data.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = data.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def type_memory(data) :
    for dtype in ['float','int','object']:
        selected_dtype = data.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

def int_memory_reduce(data) :
    data_int = data.select_dtypes(include=['int'])
    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
    data[converted_int.columns] = converted_int
    print("Change Memroy : {} == > {}".format(mem_usage(data_int) , mem_usage(converted_int)))
    return data


def float_memory_reduce(data) :
    data_float = data.select_dtypes(include=['float'])
    converted_float = data_float.apply(pd.to_numeric,downcast='float')
    print("Change Memroy : {} == > {}".format(mem_usage(data_float) , mem_usage(converted_float) ))
    data[converted_float.columns] = converted_float
    return data


def object_memory_reduce(data) :
    gl_obj = data.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            
            converted_obj.loc[:,col] = gl_obj[col].astype('category')
        else:
            print("{} 컬럼은 유니크한 숫자가 50%를 넘습니다. 체크 대상".format(col))
            converted_obj.loc[:,col] = gl_obj[col]
    print("Change Memroy : {} == > {}".format(mem_usage(gl_obj) , mem_usage(converted_obj) ))
    data[converted_obj.columns] = converted_obj
    return data
