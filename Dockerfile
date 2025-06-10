# Étape 1 : Image de base avec Python 3.10
FROM python:3.10-slim

# Étape 2 : Définir le dossier de travail
WORKDIR /app

# Étape 3 : Copier les fichiers de dépendances
COPY requirements.txt .

# Étape 4 : Mise à jour et installation des bibliothèques nécessaires (OpenCV, etc.)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Étape 5 : Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape 6 : Copier le code source de l’API
COPY api/ api/

# Étape 7 : Spécifier la variable d’environnement du port obligatoire (7860)
ENV PORT=7860

# Étape 8 : Exposer ce port
EXPOSE 7860

# Étape 9 : Commande de lancement de l'application FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
