# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 20:01:07 2022

@author: user
"""
from transformers import BatchEncoding

data = {"inputs": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]}

encod = BatchEncoding(data)

ec = encod.convert_to_tensors('pt')
print(ec)