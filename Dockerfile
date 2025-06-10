# Étape 1 : Image de base avec Python 3.10
FROM python:3.10-slim

# Étape 2 : Dossier de travail
WORKDIR /app

# Étape 3 : Copier les fichiers de dépendances
COPY requirements.txt .


# Étape 4 : Installer d'abord les dépendances lourdes (caching efficace)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0



# Étape 6 : Copier le reste du projet
COPY api/ api/
    
# Étape 7 : Définir un port d'exécution par défaut
ENV PORT=8000

# Étape 8 : Exposer le port
EXPOSE 8000

# Étape 9 : Lancer le serveur FastAPI avec Uvicorn
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

