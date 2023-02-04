---
title: 如何构建一个自定义huggingface dataset数据集？
date: 2023-02-03 12:37:34
categories:
  - AI
tags:
  - huggingface
  - python
  - dataset
  - AI
toc: true # 是否启用内容索引
---

huggingface dataset中又很多开源数据集，使用起来非常方便，加载数据集代码如下所示。

```python

from datasets import load_dataset
dataset = load_dataset("glue", "ax")

```

有时，我们希望使用自己的数据集，又与huggingface代码兼容，那就要自己构建一个dataset了。
通常我们的数据是放在csv或excel表格中，通过pandas读取，那如何把表格数据转化为dataset呢？

- csv文件或json文件，直接使用load_dataset

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("csv", data_files="my_file.csv")
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

dataset = load_dataset("json", data_files="my_file.json")
dataset = load_dataset('json', data_files={'train': 'train.json', 'test': 'test.json'})

```

- 通过DatasetDict与from_pandas分别构建
  
```python

import pandas as pd
from datasets import Dataset, DatasetDict
 

train = Dataset.from_pandas(pd.read_csv('train_spam.csv'))
test = Dataset.from_pandas(pd.read_csv('test_spam.csv'))
 
dataset = DatasetDict()
dataset['train'] = train
dataset['test'] = test

```

- 通过python的 dict、list、generator构建

```python
from datasets import Dataset

# dict
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# list
my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
dataset = Dataset.from_list(my_list)

# generator
def my_gen():
    for i in range(1, 4):
        yield {"a": i}
dataset = Dataset.from_generator(my_gen)
```
