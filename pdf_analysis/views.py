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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load OCR and Models
reader = easyocr.Reader(['en'], gpu=True)
nlp_custom = spacy.load("/home/kishore/project@sq1/Notebooks/fianl_spacy-model/model-best")
nlp_500to628 = spacy.load("/home/kishore/project@sq1/Notebooks/501to628/kaggle/working/output/model-best")
nlp_250to500 = spacy.load("/home/kishore/project@sq1/Notebooks/250to500/kaggle/working/output/model-best")
nlp = spacy.load("/home/kishore/project@sq1/Test/models 2/models 1/date_output/model-best")
nlp_person = spacy.load('/home/kishore/project@sq1/Test/models 2/models 1/name_extraction_model')
embedding_model = SentenceTransformer("/home/kishore/project@sq1/Test/models 2/models 1/sentence_transformer_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiagnosisModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiagnosisModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
   
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
input_dim = 5000  # Set the same input dimension as when training
output_dim = 14585  # Set the number of classes (update this based on your model)
desc_model = DiagnosisModel(input_dim, output_dim)
# Load the model's state_dict (weights)
desc_model.load_state_dict(torch.load('/home/kishore/project@sq1/Test/models 2/models 1/New_description_model/New_description_model/diagnosis_model.pth'))
desc_model.eval()  # Set the model to evaluation mode
code_encoder = joblib.load('/home/kishore/project@sq1/Test/models 2/models 1/New_description_model/New_description_model/label_encoder.pkl')  # Save and load the label encoder
vectorizer = joblib.load('/home/kishore/project@sq1/Test/models 2/models 1/New_description_model/New_description_model/tfidf_vectorizer.pkl')
def is_valid_description(description, threshold=0.3):
    vectorized = vectorizer.transform([description]).toarray()
    similarity = np.max(vectorized)  # Check highest TF-IDF match
    return similarity > threshold

# with open("/home/balaji/POC/POC/EasyOCR-ChatBot/models 1/models/label_encoder.pkl", "rb") as f:
#     code_encoder = pickle.load(f)

Code_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Model for Code Prediction
with open("/home/kishore/project@sq1/Test/models 2/models 1/models/code_to_idx.pkl", 'rb') as f:
    code_to_idx = pickle.load(f)

with open('/home/kishore/project@sq1/Test/models 2/models 1/models/mlb_classes.pkl', 'rb') as f:
    mlb_classes = pickle.load(f)

# Functionality to extract the name...

with open("/home/kishore/project@sq1/Test/models 2/models 1/name_embeddings.pkl", "rb") as f:
    name_embedding_dict = pickle.load(f)

known_names = list(name_embedding_dict.keys())
known_embeddings = np.array(list(name_embedding_dict.values()))

def extract_names(text):
    """Extract potential names using spaCy NER"""
    doc = nlp_person(text)
    candidates = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
    return list(set(candidates))  # Remove duplicates


def find_best_match(extracted_name):
    """Find best matching name using cosine similarity"""
    extracted_emb = embedding_model.encode([extracted_name], convert_to_numpy=True)
    similarities = cosine_similarity(extracted_emb, known_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]
    
    return known_names[best_match_index] if best_match_score >= 0.8 else extracted_name  # Return best match or original name


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
loaded_model.load_state_dict(torch.load('/home/kishore/project@sq1/Test/models 2/models 1/models/diabetes_model.pth'))
loaded_model.eval()

with open('/home/kishore/project@sq1/Test/models 2/models 1/models/label_encoder.pkl', "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Helper Functions for Text and Model Prediction

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def desc_to_code(description, desc_model, vectorizer, code_encoder, device):
    desc_model.eval()  # Set model to evaluation mode
 
    with torch.no_grad():
        # Vectorize input and convert to tensor
        vectorized = vectorizer.transform([description]).toarray()
        input_tensor = torch.tensor(vectorized, dtype=torch.float32)
 
        # Ensure model and input tensor are on the same device
        model_device = next(desc_model.parameters()).device
        input_tensor = input_tensor.to(model_device)
        desc_model.to(model_device)  # Ensure model is also on correct device
 
        # Forward pass
        outputs = desc_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
 
        # Get most probable class
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_code = code_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
 
        return predicted_code, confidence.cpu().item()
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
        return None

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

    return closest_keyword 


def process_text_with_model(text):
    doc_custom = nlp_custom(text)
    doc_bc5cdr = nlp(text)
    doc_500to628 = nlp_500to628(text)
    doc_250to500 = nlp_250to500(text)
    doc_person = nlp_person(text)
    combined_entities = set()

    # Pattern to match MM/DD/YYYY or MM/DD/YY format
    # date_pattern = r"\b(0[1-9]|1[0-2])/[0-3][0-9]/(\d{2}|\d{4})\b"
    date_pattern = r"\b(?:\d{1,2}[/]\d{1,2}[/]\d{2,4}|\d{2,4}[/]\d{1,2}[/]\d{1,2})\b"

    # Extract entities (DIAGNOSIS + properly formatted DATE + PERSON)
    for ent in doc_custom.ents:
        if ent.label_ in 'diagnosis':
            combined_entities.add((ent.text, ent.label_))
    for ent in doc_500to628.ents:
        if ent.label_== 'diagnosis':
            combined_entities.add((ent.text, ent.label_))
    for ent in doc_250to500.ents:
        if ent.label_== 'diagnosis':
            combined_entities.add((ent.text, ent.label_))

    for ent in doc_bc5cdr.ents:
        if ent.label_ == "DATE" and re.fullmatch(date_pattern, ent.text):
            date_type = classify_date(ent.text, text)  # Classify date type
            combined_entities.add((ent.text, ent.label_, date_type))  # Store date with type

    extracted_names = extract_names(text)
    for name in extracted_names:
        matched_name = find_best_match(name)  # Match with known names
        combined_entities.add((matched_name, "PERSON"))  # Store matched names
    # Predict codes for extracted entities
    # print(combined_entities)
    results = []
    for entity_info in combined_entities:
        if len(entity_info) == 2:  # Non-date entities (DIAGNOSIS, PERSON)
            entity, label = entity_info
            if label in 'diagnosis':
                predicted_code, confidence = desc_to_code(entity, desc_model, vectorizer, code_encoder, device)
                if confidence > 0.90:
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
    desc_model.eval()
    with torch.no_grad():
        # Vectorize the input description
        vectorized = vectorizer.transform([description]).toarray()
        input_tensor = torch.tensor(vectorized, dtype=torch.float32)
 
        # Move input tensor to the same device as the model
        model_device = next(desc_model.parameters()).device
        input_tensor = input_tensor.to(model_device)
 
        # Get model predictions
        outputs = desc_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
 
        # Get the most probable code
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_code = label_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
 
        return predicted_code, confidence.cpu().item()

le_combo = joblib.load('/home/kishore/project@sq1/Test/models 2/models 1/combo_code_models/label_encoder.pkl')
loaded_combo_model = joblib.load('/home/kishore/project@sq1/Test/models 2/models 1/combo_code_models/Decision_tree_model.pkl')

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
            if is_valid_description(user_input):
                response['code_from_desc'] = {'code': code, 'confidence': confidence}
            else:
                response['error'] = "Invalid description. Please provide a valid description."


        # print('Top of the Combo code....')
        elif action == 'predict_combo_code' and ',' in user_input:
            print('Inside Combo code')
            primary_code, secondary_code = user_input.split(',')
            response['combo_code'] = predict_combo_code(primary_code.strip(), secondary_code.strip())

        else:
            response['error'] = "Invalid action or missing input."
        return JsonResponse(response)

    return render(request, 'pdf_analysis/results.html')