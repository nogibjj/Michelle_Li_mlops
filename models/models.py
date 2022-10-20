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

from datasets import load_dataset, load_metric

# load data from Hugging Face
train_df, valid_df = load_dataset("imdb", split=["train", "test"])

# convert data to pandas dataframe
train_df = pd.DataFrame(train_df).sample(frac=1)[:1000]
valid_df = pd.DataFrame(valid_df).sample(frac=1)[:1000]

# text vectorization based on Tf-idf method on train data
tfvec = TfidfVectorizer(use_idf=True, min_df=2, ngram_range=(1, 2))
tdf_train = tfvec.fit_transform(train_df["text"])

# text vectorization based on Tf-idf method on valid data
tdf_valid = tfvec.transform(valid_df["text"])

# logistic regression
lr = LogisticRegression(random_state=0)

# svc
svc = LinearSVC()

# MultinomialNB
nb = MultinomialNB()
nb.fit(tdf_train, train_df["label"])

# DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(tdf_train, train_df["label"])

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

# gradio interface
def predict_tree(text):
    output = tree.predict(tfvec.transform([text]))
    print(output[0])
    return output[0]


def predict_nb(text):
    output = nb.predict(tfvec.transform([text]))
    print(output[0])
    return output[0]

def simple(text):
    output = tfvec.transform([text])
    return output

predict_tree("I love this movie")
predict_nb("I love this movie")
tree = gr.Interface(
    simple,
    gr.inputs.Textbox(label="Text"),
    gr.outputs.Textbox(label="Predicted Label"),
    capture_session=True,
    title="Example",
    description="This is the description.",
)
nb = gr.Interface(
    simple,
    inputs="textbox",
    outputs="label",
    capture_session=True,
    title="Example",
    description="This is the description.",
)

gr.Parallel(
    tree,
    nb,
    title="Compare 2 AI Summarizers",
    inputs=gr.inputs.Textbox(lines=20, label="Paste some English text here"),
)

def greet(name):
    return nb.predict(tfvec.transform([name]))
    #return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")