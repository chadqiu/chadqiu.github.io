---
title: 如何使用huggingface的trainer训练模型？
date: 2023-02-03 21:34:25
categories:
  - AI
tags:
  - huggingface
  - python
  - train
  - NLP
toc: true # 是否启用内容索引
---

huggingface上又很多开源模型，可以直接开箱即用，一个简单的模型使用实例如下：

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-8_H-512')
model = BertModel.from_pretrained("uer/chinese_roberta_L-8_H-512")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

```

有时候，我们需要finetune自己的模型，通常使用pytorch代码训练，写起来比较复杂，如果使用huggingface的trainer来训练就很方便了。

## 训练一个NLU模型

本文将使用trainer 训练一个牛客网讨论帖文本分类模型。详细过程如下：

### 构建数据集

数据集下载链接：
[train data](https://github.com/chadqiu/newcoder-crawler/blob/main/train.csv)
[test data](https://github.com/chadqiu/newcoder-crawler/blob/main/test.csv)
正常的训练演示用这两个数据集就够了，如果需要训练很精确的模型，可以使用伪标签大数据集[generated pesudo data](https://github.com/chadqiu/newcoder-crawler/blob/main/generated_pesudo_data.csv)
数据集的结构如下：
![dataset](/images/discuss_data.png)
每条数据包含一个文本和一个label，label为： [招聘信息、 经验贴、 求助贴] 三种类型之一。
我们需要加载数据集，并将文本tokenize成id，代码如下：

```python
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification

model_name = "bert-base-chinese"

max_input_length = 128
label2id = {
    '招聘信息':0,
    '经验贴':1,
    '求助贴':2
}
id2label = {v:k for k,v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, truncation=True)
    labels = [label2id[x] for x in examples['target']]
    model_inputs["labels"] = labels
    return model_inputs


raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)
```

### 定义评价指标函数

评价指标metric用于evaluate的时候衡量模型的表现，这里使用f1 score 和 accuracy

```python
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import EvalPrediction

def multi_label_metrics(predictions, labels, threshold=0.5):
    probs =  np.argmax( predictions, -1)       
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=probs, average='micro')
    accuracy = accuracy_score(y_true, probs)
    print(classification_report([id2label[x] for x in y_true], [id2label[x] for x in probs]))
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics
 
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result
```

### 指定模型的训练参数

加载模型，并构建TrainingArguments类，用于指定模型训练的各种参数
第一个是训练保存地址为必填项，其他都是选填项

```python
from transformers import TrainingArguments, Trainer

batch_size = 64

training_args = TrainingArguments(
    f"/root/autodl-tmp/run",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # gradient_accumulation_steps=2,
    num_train_epochs=10,
    save_total_limit=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    fp16=True,
)

```

### 定义trainer并进行训练

```python
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()  # 开始训练

```

### 测试预测

```python
print("test")
print(trainer.evaluate())  # 测试
trainer.save_model("bert")  #保存模型

# 进行模型预测，并将预测结果输出便于观察
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=-1)
print(predictions)
print(labels)

```

### 代码整合

将上面代码整合到一起，结果如下：

```python
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from transformers import EvalPrediction

import evaluate

metric = evaluate.load("seqeval")

model_name = "uer/chinese_roberta_L-4_H-512"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_length = 128
label2id = {
    '招聘信息':0,
    '经验贴':1,
    '求助贴':2
}
id2label = {v:k for k,v in label2id.items()}

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, truncation=True)
    labels = [label2id[x] for x in examples['target']]
    model_inputs["labels"] = labels
    return model_inputs

raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)


def multi_label_metrics(predictions, labels, threshold=0.5):
    probs =  np.argmax( predictions, -1)       
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=probs, average='micro')
    accuracy = accuracy_score(y_true, probs)
    print(classification_report([id2label[x] for x in y_true], [id2label[x] for x in probs]))
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics
 
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                        # problem_type="multi_label_classification", 
                                        num_labels=3,
                                        # id2label=id2label,
                                        # label2id=label2id
                                        )

batch_size = 64
metric_name = "f1"

training_args = TrainingArguments(
    f"/root/autodl-tmp/run",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # gradient_accumulation_steps=2,
    num_train_epochs=10,
    save_total_limit=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    fp16=True,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("test")
print(trainer.evaluate())
trainer.save_model("bert")

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=-1)

print(predictions)
print(labels)

```

### 模型推理预测

使用训练好的模型在其他数据集上推理预测，新数据集是从牛客网爬取的帖子信息,接近4万条，数据链接： [historical_data](https://github.com/chadqiu/newcoder-crawler/blob/main/historical_data.xlsx)
数据截图如下：
![historical_data](/images/newcoder_data.png)

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

data = pd.read_excel("historical_data.xlsx", sheet_name=0).fillna(" ")
data['text'] = data['title'].apply(lambda x : str(x) if x else "") + data['content'].apply(lambda x : str(x) if x else "")

model_name = "bert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    device = "cuda:0"
    model.half()
else:
    device = "cpu"
model = model.to(device)

max_target_length = 128
label2id = {
    '招聘信息':0,
    '经验贴':1,
    '求助贴':2
}
id2label = {v:k for k,v in label2id.items()}

def get_answer(text):
    text = [x for x in text]
    inputs = tokenizer( text, return_tensors="pt", max_length=max_target_length, padding=True, truncation=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs).logits.argmax(-1).tolist()
    return outputs

# print(get_answer(data['text'][:10]))

pred , grod = [], []
index, batch_size = 0, 32

while index < len(data['text']):
    pred.extend(get_answer([x for x in data['text'][index:index + batch_size]]))
    index += batch_size

# print(pred)
# print(grod)

pred = [id2label[x] for x in pred]
data["target"] = pred

writer = pd.ExcelWriter("generate.xlsx")
data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
writer.save()
writer.close()

```

## 训练seq2seq生成式模型T5

上面的例子是判别式模型，只用到了encoder，接下来训练一个encoder-decoder base的生成式模型T5，使用prompt用于训练，prompt方式如下：

```python

input:
请问下面文本属于哪一类帖子？
秋招大结局（泪目了）。家人们泪目了，一波三折之后获得的小奖状，已经准备春招了，没想到被捞啦，嗐，总之是有个结果，还是很开心的[掉小珍珠了][掉小珍珠了]
选项：招聘信息, 经验贴, 求助贴
答案：

output:
经验贴
```

### 构建数据集

```python
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

model_name = "ClueAI/ChatYuan-large-v1"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

max_input_length = 128
max_target_length = 20
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："

def preprocess_function(examples):
    inputs = [prefix + doc + suffix for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

```

### 等一评价指标

这次使用不一样的方式来构建评价指标

```python
import evaluate
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = [tokenizer.batch_decode(predictions, skip_special_tokens=True)] 
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.batch_decode(labels, skip_special_tokens=True)] 
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

```

### 构建trainer训练

```python
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 4

args = Seq2SeqTrainingArguments(
    f"yuan-finetuned-xsum",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 10,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    # fp16=True,
    # push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print("test")
print(trainer.evaluate())

```

### 代码整合

```python

import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import evaluate
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = [tokenizer.batch_decode(predictions, skip_special_tokens=True)] 
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.batch_decode(labels, skip_special_tokens=True)] 
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

model_name = "ClueAI/ChatYuan-large-v1"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

max_input_length = 252
max_target_length = 20
batch_size = 4
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："

def preprocess_function(examples):
    inputs = [prefix + doc + suffix for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

args = Seq2SeqTrainingArguments(
    f"yuan-finetuned-yuan",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 10,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print("test")
print(trainer.evaluate())
trainer.save_model("yuan")

```

### 模型推理预测

```python
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import pandas as pd
import torch

data = pd.read_excel("historical_data.xlsx", sheet_name = 0).fillna(" ")
data['text'] = data['title'].apply(lambda x : str(x) if x else "") + data['content'].apply(lambda x : str(x) if x else "")

model_name = "yuan"
max_target_length = 512
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    device = "cuda:0"
    model.half()
else:
    device = "cpu"
model = model.to(device)

prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："

def get_answer(text):
    if not text :
        return ""
    inputs = tokenizer( prefix + str(text) + suffix, return_tensors="pt", max_length=max_target_length, truncation=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    # print(inputs)
    outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True)
    return tokenizer.decode(outputs[0][0], skip_special_tokens=True)

data['target'] = data['text'].map(get_answer)  # not recommend, it's better to generate in batches 

writer = pd.ExcelWriter("generate.xlsx")
data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
writer.save()
writer.close()

```