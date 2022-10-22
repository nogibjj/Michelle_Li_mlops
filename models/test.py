import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer

import gradio as gr
from gradio.mix import Parallel

from datasets import load_dataset

# load data from Hugging Face
train_df, valid_df = load_dataset("imdb", split=["train", "test"])

# convert data to pandas dataframe
train_df = pd.DataFrame(train_df).sample(frac=1)[:10000]
valid_df = pd.DataFrame(valid_df).sample(frac=1)[:1000]

# split into text and labels
x_train = train_df["text"]
y_train = train_df["label"]

# vectorize text
tfidf_vect = TfidfVectorizer(use_idf=True)
x_train_tfidf = tfidf_vect.fit_transform(x_train)

# fit models

# DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0).fit(x_train_tfidf, y_train)

# svc
svc = LinearSVC().fit(x_train_tfidf, y_train)

# MultinomialNB
nb = MultinomialNB().fit(x_train_tfidf, y_train)

# logistic regression
lr = LogisticRegression(random_state=0).fit(x_train_tfidf, y_train)

# Hugging Face transformer model
pipe = pipeline("text-classification", model='bert-base-cased', binary_output = True)

def transfomer_predict(text):
  x = pipe(text)[0]["label"]
  if x == "LABEL_0":
    return 0
  else:
    return 1

def tree_predict(text):
    text = [text]
    tfidf = tfidf_vect.transform(text)
    output = tree.predict(tfidf)
    return output[0]

def svc_predict(text):
    text = [text]
    tfidf = tfidf_vect.transform(text)
    output = svc.predict(tfidf)
    return output[0]


def nb_predict(text):
    text = [text]
    tfidf = tfidf_vect.transform(text)
    output = nb.predict(tfidf)
    return output[0]


def lr_predict(text):
    text = [text]
    tfidf = tfidf_vect.transform(text)
    output = lr.predict(tfidf)
    return output[0]

iface = gr.Interface(
    fn=transfomer_predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Transformer Classifier"),
)

tree_gui = gr.Interface(
    fn=tree_predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Decision Tree Classifier"),
)
svc_gui = gr.Interface(
    fn=svc_predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Support Vector Classifier"),
)
nb_gui = gr.Interface(
    fn=nb_predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Naive Bayes Classifier"),
)
lr_gui = gr.Interface(
    fn=lr_predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(label="Logistic Regression Classifier"),
)
Parallel(
    iface,
    tree_gui,
    svc_gui,
    nb_gui,
    lr_gui,
    title="Compare Text Classification Machine Learning Models",
    inputs=gr.inputs.Textbox(lines=20, label="Paste some English text here"),
).launch()