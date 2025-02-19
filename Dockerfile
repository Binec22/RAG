# Utiliser une image de base officielle Python
FROM python:3.12.6

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va tourner
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "local_app.py"]
