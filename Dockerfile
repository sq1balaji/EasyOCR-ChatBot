# Use the official Python image
FROM python:3.12.3

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y poppler-utils

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the Django project files
COPY . .

# Copy the models explicitly (if they are ignored by .gitignore)
COPY pdf_analysis/models/ /app/pdf_analysis/models/


# Expose port 8000 for Django
EXPOSE 8000

# Run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
