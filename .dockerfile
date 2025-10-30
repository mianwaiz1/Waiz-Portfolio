# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Environment Variables ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Dependencies ----------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------- Copy Project Files ----------
COPY . .

# ---------- Expose Flask Port ----------
EXPOSE 5000

# ---------- Environment ----------
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# ---------- Run App ----------
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
