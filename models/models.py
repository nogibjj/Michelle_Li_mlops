"""Compare different types of models."""
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import gradio as gr

from datasets import load_dataset

# load data from Hugging Face
train_df, valid_df = load_dataset("imdb", split=["train", "test"])

# convert data to pandas dataframe
train_df = pd.DataFrame(train_df).sample(frac=1)[:10000]
valid_df = pd.DataFrame(valid_df).sample(frac=1)[:1000]

# split into text and labels
x_train = train_df["text"]
y_train = train_df["label"]

#split into text and labels
x_valid = valid_df["text"]
y_valid = valid_df["label"]

# text vectorization based on Tf-idf method on train data
tfvec = TfidfVectorizer(use_idf=True)
tdf_train = tfvec.fit_transform(x_train)

# text vectorization based on Tf-idf method on valid data
tdf_valid = tfvec.transform(x_valid)

# logistic regression
lr = LogisticRegression(random_state=0).fit(tdf_train, y_train)

# svc
svc = LinearSVC().fit(tdf_train, y_train)

# MultinomialNB
nb = MultinomialNB().fit(tdf_train, y_train)

# DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0).fit(tdf_train, y_train)

# train model
def train_model(model, train_data, train_labels, valid_data, valid_labels):
    """Train model."""
    model.fit(train_data, train_labels)
    valid_predict = model.predict(valid_data)
    valid_matrix = confusion_matrix(valid_labels, valid_predict)
    print("confusion matrix:", valid_matrix)
    print("accuracy:", sklearn.metrics.accuracy_score(valid_labels, valid_predict))
    print(
        "precision:",
        sklearn.metrics.precision_score(valid_labels, valid_predict, average="macro"),
    )
    print(
        "recall:",
        sklearn.metrics.recall_score(valid_labels, valid_predict, average="macro"),
    )
    print("f1:", sklearn.metrics.f1_score(valid_labels, valid_predict, average="macro"))
    return model


# compare models on valid data
def compare_models(train_data, train_labels, valid_data, valid_labels):
    """Compare models."""
    print("logistic regression")
    train_model(lr, train_data, train_labels, valid_data, valid_labels)
    print("svc")
    train_model(svc, train_data, train_labels, valid_data, valid_labels)
    print("nb")
    train_model(nb, train_data, train_labels, valid_data, valid_labels)
    print("tree")
    train_model(tree, train_data, train_labels, valid_data, valid_labels)


compare_models(tdf_train, train_df["label"], tdf_valid, valid_df["label"])