import gradio as gr

from transformers import pipeline

pipe = pipeline("text-classification", model="bert-base-cased")

classify_pipeline = pipeline("text-classification")

examples = [
    "Does Chicago have any stores and does Joe live here?",
]

def predict(text):
    output = classify_pipeline(text)
    return output 

demo = gr.Interface(fn=predict,inputs="text",outputs="label").launch()

demo.launch()