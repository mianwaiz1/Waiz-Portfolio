# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Environment Setup ----------
# Prevent Python from writing pyc files and using output buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# ---------- System Dependencies ----------
# Install system packages required by PyMuPDF, FAISS, and LangChain
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Project Files ----------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire application
COPY . .

# ---------- Security & Flask Settings ----------
# Flask will run on port 5000
EXPOSE 5000

# Disable Flask’s debug mode in production by default
ENV FLASK_DEBUG=false
ENV FLASK_APP=app.py

# ---------- Run the App ----------
# Using gunicorn for production (better than flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
