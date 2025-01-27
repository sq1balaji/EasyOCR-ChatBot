import os
import time
from django.shortcuts import render
from .forms import PDFUploadForm
from .models import UploadedPDF
from pdf2image import convert_from_path
import numpy as np
import easyocr
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
import pickle
import torch.nn.functional as F
import re
from django.conf import settings  # Import the settings module
from django.http import JsonResponse
import torch.nn as nn
from django.views.decorators.csrf import csrf_exempt
import joblib
import hashlib
import torch.nn.functional as F

# Load OCR and Models
reader = easyocr.Reader(['en'], gpu=True)
nlp_custom = spacy.load("/home/kishoreb/project@sq1/POC_/POC_SQ1/predict/Spacy-Models/model-best")
nlp = spacy.load("/home/kishoreb/project@sq1/POC_/POC_SQ1/predict/Spacy-Models/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.5.4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
desc_model.load_state_dict(torch.load("/home/kishoreb/project@sq1/POC_/POC_SQ1/predict/models/final_model.pth", map_location=device))
desc_model.to(device)

with open("/home/kishoreb/project@sq1/POC_/POC_SQ1/predict/models/label_encoder.pkl", "rb") as f:
    code_encoder = pickle.load(f)
Code_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

def process_text_with_model(text):
    doc_custom = nlp_custom(text)
    doc_bc5cdr = nlp(text)
    combined_entities = set()

    # Extract entities
    for ent in doc_custom.ents:
        combined_entities.add((ent.text, ent.label_))
    for ent in doc_bc5cdr.ents:
        if ent.label_ == 'DISEASE':
            combined_entities.add((ent.text, ent.label_))

    # Predict codes for extracted entities
    results = []
    for entity, label in combined_entities:
        if label == "DISEASE":
            predicted_code, confidence = desc_to_code(entity, desc_model, Code_Tokenizer, code_encoder, device)
            results.append((entity, predicted_code, confidence))
    return results

def process_page(image, page_number):
    print(f"Processing page {page_number}...")

    # Record the start time for OCR
    start_time = time.time()
    
    # Perform OCR on the image
    image_np = np.array(image)
    results = reader.readtext(image_np)
    page_text = ' '.join([result[1] for result in results])
    page_text = clean_text(page_text)
    
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

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_instance = form.save()
            pdf_path = pdf_instance.file.path

            # Use MEDIA_URL to get the file path for the uploaded PDF
            pdf_url = pdf_instance.file.url  # This gives the relative URL
            print(pdf_url)
            
            results = []

            # Process the uploaded PDF
            images = convert_from_path(pdf_path, dpi=200)
            for i, image in enumerate(images, 1):
                page_text, model_results = process_page(image, i)
                results.append((i, page_text, model_results))

            # Pass pdf_url to the template so it can be used to display the PDF
            return render(request, 'pdf_analysis/results.html', {'results': results, 'pdf_url': pdf_url})

    else:
        form = PDFUploadForm()
    return render(request, 'pdf_analysis/upload.html', {'form': form})
