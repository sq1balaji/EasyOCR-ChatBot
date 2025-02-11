import os
import time
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import PDFUploadForm
from .models import UploadedPDF
from pdf2image import convert_from_path
import numpy as np
import easyocr
import torch
import spacy
import pickle
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import re
import hashlib
import joblib
import torch.nn as nn

# Load OCR and Models
reader = easyocr.Reader(['en'], gpu=True)
nlp_custom = spacy.load("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/Spacy-Models/model-best")
nlp = spacy.load("/home/balaji/POC/POC/EasyOCR-ChatBot/output/model-best")
nlp_person = spacy.load('en_core_web_sm')
# nlp = spacy.load("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/Spacy-Models/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.5.4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# x
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
desc_model.load_state_dict(torch.load("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/final_model.pth", map_location=device))
desc_model.to(device)

with open("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/label_encoder.pkl", "rb") as f:
    code_encoder = pickle.load(f)

Code_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Model for Code Prediction
with open("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/code_to_idx.pkl", 'rb') as f:
    code_to_idx = pickle.load(f)

with open('/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/mlb_classes.pkl', 'rb') as f:
    mlb_classes = pickle.load(f)

num_codes = len(code_to_idx)
num_labels = len(mlb_classes)


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


loaded_model = MultiLabelModel(num_codes, num_labels)
loaded_model.load_state_dict(torch.load('/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/diabetes_model.pth'))
loaded_model.eval()

with open('/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/label_encoder.pkl', "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Helper Functions for Text and Model Prediction

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def desc_to_code(description, desc_model, Code_Tokenizer, code_encoder, device, max_length=128):
    with torch.no_grad():
        encoded = Code_Tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = desc_model(input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_code = code_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
        return predicted_code, confidence.cpu().item()

""" Old Views of process_text_with_model"""
# import re

# def process_text_with_model(text):
#     doc_custom = nlp_custom(text)
#     doc_bc5cdr = nlp(text)
#     doc_person = nlp_person(text)
#     combined_entities = set()

#     # Pattern to match MM/DD/YYYY or MM/DD/YY fromat
#     date_pattern = r"\b(0[1-9]|1[0-2])/[0-3][0-9]/(\d{2}|\d{4})\b"

#     # Extract entities (DIAGNOSIS + properly formatted DATE)
#     for ent in doc_custom.ents:
#         if ent.label_ in ["DIAGNOSIS", "DISEASE"]:
#             combined_entities.add((ent.text, ent.label_))

#     for ent in doc_bc5cdr.ents:
#         if ent.label_ == "DATE" and re.fullmatch(date_pattern, ent.text):
#             combined_entities.add((ent.text, ent.label_))

#     for ent in doc_person.ents:
#         if ent.label_ == "PERSON":
#             combined_entities.add((ent.text , ent.label_))

#     # Predict codes for extracted entities
#     results = []
#     for entity, label in combined_entities:
#         if label in ["DIAGNOSIS", "DISEASE" , "DATE" , "PERSON"]:  # Predict code only for DIAGNOSIS
#             predicted_code, confidence = desc_to_code(entity, desc_model, Code_Tokenizer, code_encoder, device)
#             results.append((entity, predicted_code, confidence, label))
        
#         # else:  # DATE does not need a prediction
#         #     results.append((entity, "", "", label))  

#     return results

""" Updated Views of process_text_with_model"""
import re

def classify_date(entity_text, text):
    """
    Classifies a date as Date of Birth (DOB), Visit Date, Exam Date, etc.,
    by searching for contextual keywords within a 50-character range.
    """
    text_lower = text.lower()
    entity_text_lower = entity_text.lower()

    # Define contextual keywords
    keyword_labels = {
        "dob": "Date of Birth",
        "date of birth": "Date of Birth",
        "born": "Date of Birth",
        "birthdate": "Date of Birth",
        "exam date": "Exam Date",
        "service date": "Service Date/Time",
        "service date/time": "Service Date/Time",
        "triage date": "Triage Date",
        "visit date": "Visit Date",
        "appointment": "Visit Date",
        "consultation": "Visit Date",
        "seen on": "Visit Date",
        "visited": "Visit Date",
        "visit note": "Visit Date",
        "visited note": "Visit Date",
        "encounter date": "Encounter Date",
        "encountered date": "Encounter Date",
    }

    # Find the position of the date in text
    match = re.search(re.escape(entity_text_lower), text_lower)
    if not match:
        return "Uncategorized Date"

    date_start, date_end = match.start(), match.end()

    # Define search range (50 characters before and after)
    search_start = max(0, date_start - 50)
    search_end = min(len(text_lower), date_end + 50)
    search_text = text_lower[search_start:search_end]

    # Search for the closest keyword within the range
    closest_keyword = None
    min_distance = float('inf')

    for keyword, label in keyword_labels.items():
        for match in re.finditer(re.escape(keyword), search_text):
            keyword_start, keyword_end = match.start(), match.end()
            
            # Convert keyword positions to original text positions
            keyword_start += search_start
            keyword_end += search_start
            
            # Calculate distance
            if keyword_end <= date_start:  # Keyword is before date
                distance = date_start - keyword_end
            elif keyword_start >= date_end:  # Keyword is after date
                distance = keyword_start - date_end
            else:
                continue  # Ignore if keyword overlaps the date

            # Assign closest keyword
            if distance < min_distance:
                min_distance = distance
                closest_keyword = label

    return closest_keyword if closest_keyword else "Uncategorized Date"



def process_text_with_model(text):
    doc_custom = nlp_custom(text)
    doc_bc5cdr = nlp(text)
    doc_person = nlp_person(text)
    combined_entities = set()

    # Pattern to match MM/DD/YYYY or MM/DD/YY format
    # date_pattern = r"\b(0[1-9]|1[0-2])/[0-3][0-9]/(\d{2}|\d{4})\b"
    date_pattern = r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(?:\d{2}|\d{4})\b"

    # Extract entities (DIAGNOSIS + properly formatted DATE + PERSON)
    for ent in doc_custom.ents:
        if ent.label_ in ["DIAGNOSIS", "DISEASE"]:
            combined_entities.add((ent.text, ent.label_))

    for ent in doc_bc5cdr.ents:
        if ent.label_ == "DATE" and re.fullmatch(date_pattern, ent.text):
            date_type = classify_date(ent.text, text)  # Classify date type
            combined_entities.add((ent.text, ent.label_, date_type))  # Store date with type

    for ent in doc_person.ents:
        if ent.label_ == "PERSON":
            combined_entities.add((ent.text, ent.label_))

    # Predict codes for extracted entities
    results = []
    for entity_info in combined_entities:
        if len(entity_info) == 2:  # Non-date entities (DIAGNOSIS, PERSON)
            entity, label = entity_info
            if label in ["DIAGNOSIS", "DISEASE"]:
                predicted_code, confidence = desc_to_code(entity, desc_model, Code_Tokenizer, code_encoder, device)
                results.append((entity, predicted_code, confidence, label))
            else:
                results.append((entity, "N/A", "N/A", label))

        elif len(entity_info) == 3:  # Date entities (DATE + classified type)
            entity, label, date_type = entity_info
            results.append((entity, "N/A", "N/A", label, date_type))

    return results

# Function to process OCR and model analysis

def process_page(image, page_number):
    print(f"\nProcessing page {page_number}...")

    # Record the start time for OCR
    start_time = time.time()
    
    # Perform OCR on the image
    image_np = np.array(image)
    results = reader.readtext(image_np)
    page_text = ' '.join([result[1] for result in results])
    page_text = clean_text(page_text)
    print(page_text)
    
    # Calculate and print time taken for OCR
    ocr_time = time.time() - start_time
    print(f"Time taken for OCR on page {page_number}: {ocr_time:.2f} seconds")

    # Record the start time for model analysis
    start_time = time.time()
    
    # Perform model analysis
    model_results = process_text_with_model(page_text)
    
    # Calculate and print time taken for analysis
    analysis_time = time.time() - start_time
    print(f"Time taken for analysis on page {page_number}: {analysis_time:.2f} seconds")
    
    return page_text, model_results

# PDF Upload Handling

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_instance = form.save()
            pdf_path = pdf_instance.file.path

            # Use MEDIA_URL to get the file path for the uploaded PDF
            pdf_url = pdf_instance.file.url  # This gives the relative URL
            # print(pdf_url)
            
            results = []

            # Process the uploaded PDF
            images = convert_from_path(pdf_path, dpi=500)
            for i, image in enumerate(images, 1):
                page_text, model_results = process_page(image, i)
                results.append((i, page_text, model_results))

            # Pass pdf_url to the template so it can be used to display the PDF
            return render(request, 'pdf_analysis/results.html', {'results': results, 'pdf_url': pdf_url})

    else:
        form = PDFUploadForm()
    return render(request, 'pdf_analysis/upload.html', {'form': form})

# Functions for predictions

def predict_descriptions(user_code):
    # print('Inside Func....')
    user_code = user_code.upper()
    if user_code not in code_to_idx:
        return [f"No descriptions predicted for code '{user_code}'."]
    
    code_idx = torch.tensor([code_to_idx[user_code]], dtype=torch.long)
    output = loaded_model(code_idx)
    output = (output > 0.1).float()
    predicted_descriptions = [desc for idx, desc in enumerate(mlb_classes) if output[0, idx] == 1]
    return predicted_descriptions if predicted_descriptions else [f"No descriptions predicted for code '{user_code}'."]

def predict_code(description):
    encoded = tokenizer(description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = desc_model(input_ids, attention_mask)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_code = label_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
    return predicted_code, confidence.cpu().item()

le_combo = joblib.load('/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/combo_code_models/label_encoder.pkl')
loaded_combo_model = joblib.load('/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/combo_code_models/Decision_tree_model.pkl')

def predict_combo_code(primary_code, secondary_code):
    primary_code = primary_code.strip().upper()
    secondary_code = secondary_code.strip().upper()

    # print('Inside Combo code func...')

    # Try encoding the codes, return error if they are unknown
    try:
        primary_code_encoded = le_combo.transform([primary_code])[0]
        secondary_code_encoded = le_combo.transform([secondary_code])[0]
    except ValueError:
        return f"Error: One or both of the codes '{primary_code}' or '{secondary_code}' are not recognized."

    # Prepare input feature (model expects two variations)
    X_new = np.array([[primary_code_encoded, secondary_code_encoded], 
                      [secondary_code_encoded, primary_code_encoded]])

    # Predict combo code
    prediction_encoded = loaded_combo_model.predict(X_new)
    prediction = le_combo.inverse_transform([prediction_encoded[0]])  # Extract first value

    return prediction[0]

# Chatbot functionality for predictions

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        user_input = request.POST.get('user_input', '').strip()
        response = {}

        if action == 'predict_description' and user_input:
            response['descriptions'] = predict_descriptions(user_input)

        elif action == 'predict_code' and user_input:
            code, confidence = predict_code(user_input)
            response['code_from_desc'] = {'code': code, 'confidence': confidence}

        # print('Top of the Combo code....')
        elif action == 'predict_combo_code' and ',' in user_input:
            print('Inside Combo code')
            primary_code, secondary_code = user_input.split(',')
            response['combo_code'] = predict_combo_code(primary_code.strip(), secondary_code.strip())

        else:
            response['error'] = "Invalid action or missing input."
        return JsonResponse(response)

    return render(request, 'pdf_analysis/results.html')