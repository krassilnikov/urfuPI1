import io
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
st.title('Определятор токсичности текста')

text_from_st = st.text_input('Текст') # ввод пользователя

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

 
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors='pt', truncation=True, padding=True).to(
                model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba


def m(Text):
    st.write('Токсичность (чем больше, тем токсичнее) :', text2toxicity(Text))


m(text_from_st)
