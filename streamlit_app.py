import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
import streamlit as st

st.title('Определение семантики текста')
text_from_st = st.text_input('Текст')

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)
if torch.cuda.is_available():
    model.cuda()  
    
@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_label = torch.argmax(predicted_probabilities).item()  # Convert to a single int value
    return predicted_label, predicted_probabilities


def to_model(Text):
    LABELS = ['без эмоций', 'радость', 'грусть', 'сюрприз', 'страх', 'злость']
    predicted_label, predicted_probabilities = predict(Text)

    top_emotions = torch.argsort(predicted_probabilities, descending=True)
    thresholds = [0.90, 0.75, 0.50, 0.25]
    top_emotions_with_percentage = []
    for i in top_emotions:
        emotion = LABELS[i]
        percentage = predicted_probabilities[i].item() * 100
        if percentage >= thresholds[0]:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[1] and len(top_emotions_with_percentage) < 2:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[2] and len(top_emotions_with_percentage) < 3:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[3] and len(top_emotions_with_percentage) < 4:
            top_emotions_with_percentage.append((emotion, percentage))

    st.write('Распознанные емоции:')
    for emotion, percentage in top_emotions_with_percentage:
        st.write(f"{emotion}: {percentage:.2f}%")

to_model(text_from_st)