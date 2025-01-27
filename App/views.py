import torch
import pickle
from django.http import JsonResponse
from django.shortcuts import render
import torch.nn as nn
from django.views.decorators.csrf import csrf_exempt
import joblib
import numpy as np
import hashlib
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# model for Description prediction.

class MultiLabelModel(nn.Module):
    def __init__(self, num_codes, num_labels):
        super(MultiLabelModel, self).__init__()
        self.embedding = nn.Embedding(num_codes, 16)
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

with open('/home/balaji/project/ChatBot/Project/App/Models/models/code_to_idx.pkl', 'rb') as f:
    code_to_idx = pickle.load(f)

with open('/home/balaji/project/ChatBot/Project/App/Models/models/mlb_classes.pkl', 'rb') as f:
    mlb_classes = pickle.load(f)

num_codes = len(code_to_idx)
num_labels = len(mlb_classes)

loaded_model = MultiLabelModel(num_codes, num_labels)
loaded_model.load_state_dict(torch.load('/home/balaji/project/ChatBot/Project/App/Models/models/diabetes_model.pth'))
loaded_model.eval()

with open('/home/balaji/project/ChatBot/Project/App/Models/models/label_encoder.pkl', "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model for Code prediction.

class CodePredictionModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(CodePredictionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

desc_model = CodePredictionModel(num_labels=14585)
desc_model.load_state_dict(torch.load('/home/balaji/project/ChatBot/Project/App/Models/models/final.pth', map_location=torch.device('cpu')))
desc_model.to(device)

loaded_combo_model = joblib.load('/home/balaji/project/ChatBot/Project/App/Models/Combo_Code/Random_forest_model.pkl')
le_combo = joblib.load('/home/balaji/project/ChatBot/Project/App/Models/Combo_Code/label_encoder_combo.pkl')

# Code for decription prediction.

def predict_descriptions(user_code):
    print('Inside Func....')
    user_code = user_code.upper()
    print(user_code)
    print('Out of If....')
    if user_code not in code_to_idx:
        print('Inside If....')
        return [f"No descriptions predicted for code '{user_code}'."]
    code_idx = torch.tensor([code_to_idx[user_code]], dtype=torch.long)
    output = loaded_model(code_idx)
    output = (output > 0.1).float()
    predicted_descriptions = [desc for idx, desc in enumerate(mlb_classes) if output[0, idx] == 1]
    return predicted_descriptions if predicted_descriptions else [f"No descriptions predicted for code '{user_code}'."]

# Code for code prediction.

def predict_code(description):
    # print('Inside Func....')
    encoded = tokenizer(description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = desc_model(input_ids, attention_mask)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_code = label_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
    return predicted_code, confidence.cpu().item()

# Code for combination code prediction.

def predict_combo_code(primary_code, secondary_code):
    primary_code = primary_code.strip().upper()
    secondary_code = secondary_code.strip().upper()
    if primary_code not in le_combo.classes_ or secondary_code not in le_combo.classes_:
        return f"Error: One or both of the codes '{primary_code}' or '{secondary_code}' are not recognized."
    primary_code_encoded = le_combo.transform([primary_code])[0]
    secondary_code_encoded = le_combo.transform([secondary_code])[0]
    combined_code = tuple(sorted([primary_code_encoded, secondary_code_encoded]))
    combined_code_hashed = int(hashlib.sha256(str(combined_code).encode()).hexdigest(), 16) % (10 ** 8)
    X_new = np.array([combined_code_hashed]).reshape(-1, 1)
    prediction_encoded = loaded_combo_model.predict(X_new)
    prediction = le_combo.inverse_transform(prediction_encoded)
    return prediction[0]

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        user_input = request.POST.get('user_input', '').strip()
        response = {}

        # if action == 'predict_description' and user_input:
        if  action == 'predict_description' and user_input:
            response['descriptions'] = predict_descriptions(user_input)

        elif action == 'predict_code' and user_input:
            code, confidence = predict_code(user_input)
            response['code_from_desc'] = {'code': code, 'confidence': confidence}

        elif action == 'predict_combo_code' and ',' in user_input:
            primary_code, secondary_code = user_input.split(',')
            response['combo_code'] = predict_combo_code(primary_code.strip(), secondary_code.strip())

        else:
            response['error'] = "Invalid action or missing input."
        return JsonResponse(response)

    return render(request, 'chatbot/chat.html')
