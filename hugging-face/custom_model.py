#!/usr/bin/env python

"""
Create a custom model architecture with HuggingFace
"""

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
# configuration = model’s specific attributes
from transformers import AutoConfig, BertConfig

# laod model architecture
config = AutoConfig.from_pretrained("bert-base-cased")
print(config)

my_config = BertConfig.from_pretrained("bert-base-cased", attention_probs_dropout_prob=0.4)
print(my_config)

# save custom model architecutre in json file
my_config.save_pretrained(save_directory="./custom_bert")

# load model with my custom architecture
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", config=my_config)
