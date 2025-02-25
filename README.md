# Le Chat by Seatech

Le **Chat by Seatech** est un chatbot utilisant le **RAG** (Retrieval-Augmented Generation) pour répondre aux questions des utilisateurs concernant l'école d'ingénieurs **Seatech**. À terme, ce chatbot pourrait être intégré au site internet de l'école pour offrir une assistance interactive et informative.

---

## Fonctionnalités

- **Base de données vectorielle** : Les informations textuelles du site de Seatech sont stockées dans une base de données vectorielle, permettant de trouver instantanément la page web ou la réponse adéquate à l'utilisateur.
- **Backend en Python** : Utilisation des agents de **LangChain** pour la gestion des interactions.
- **Frontend en Flask** : Le site internet est géré par **Flask**, avec une structure en **HTML/CSS/JS pur**.

---

## Installation

### Cloner le dépôt :
```bash
git clone https://github.com/Binec22/RAG.git
cd RAG
```

### Installer les dépendances :
```bash
pip install -r requirements.txt
```

### Configurer les clés API :
1. Créez un fichier **.env** à la racine du projet.
2. Ajoutez vos clés API pour les services implémentés :

```plaintext
GROQ_API_KEY=votre_cle_groq
HF_API_KEY=votre_cle_huggingface
OPENAI_API_KEY=votre_cle_openai
VOYAGE_API_KEY=votre_cle_voyage
```

> **Note** :
> Vous n'êtes pas obligés de renseigner les clés d'API que vous ne voulez pas utiliser




### Installer Ollama (si nécessaire) :
Suivez les instructions d'installation d'**Ollama** pour utiliser les modèles locaux.

---

## Utilisation

### Configurer les paramètres :
Modifiez les paramètres dans le fichier **local_app.py** selon vos besoins :

```python
default_params = {
    "database_name": 'seatech_firecrawl',
    "temperature": 0,
    "search_type": "similarity",
    "similarity_doc_nb": 5,
    "score_threshold": 0.8,
    "max_chunk_return": 5,
    "considered_chunk": 25,
    "mmr_doc_nb": 5,
    "lambda_mult": 0.25,
    "isHistoryOn": True,
    "embedding_model": "nomic-embed-text",
    "llm_model": "voyage-3",
}
```

### Lancer le serveur Flask :
```bash
python local_app.py
```

### Accéder au site :
Ouvrez votre navigateur et accédez à **http://localhost:5000**.

---

## Gestion de la Base de Données Vectorielle

Le script de gestion de la base de données permet de **créer, réinitialiser et effacer** la base de données vectorielle utilisée par le chatbot. Voici comment l'utiliser :

### Prérequis
- Assurez-vous que toutes les dépendances sont installées (**requirements.txt**).
- Configurez les clés API nécessaires dans un fichier **.env**.

### Utilisation du Script
Le script utilise l'interface en ligne de commande (**CLI**) pour interagir avec la base de données.

#### Créer ou peupler la base de données :
```bash
python path/to/Datbase.py --config your_config_name
```
**--config** : Spécifie le nom de configuration contenant les paramètres pour initialiser la base de données. Les différentes configurations peuvent être renseignées dans le fichier JSON config.json.

#### Réinitialiser la base de données :
```bash
python path/to/Database.py --config your_config_name --reset
```
**--reset** : Réinitialise la base de données en supprimant les données existantes et en la recréant avec les documents spécifiés.

#### Effacer la base de données :
```bash
python path/to/Database.py --clear
```
**--clear** : Supprime toutes les données de la base de données sans la recréer.

---

## Exemple de Fichier de Configuration

Voici un exemple de fichier de configuration **JSON** (your_config_file.json) :

```json
{
  "your_database_name": {
    "embedding_model": "nomic-embed-text",
    "data_files_path": "path/to/your/documents",
    "embedded_database_path": "path/to/your/embedded_database"
  }
}
```

- **embedding_model** : Le modèle d'embedding utilisé pour vectoriser les documents.
- **data_files_path** : Le chemin vers les documents à indexer.
- **embedded_database_path** : Le chemin où la base de données vectorielle sera stockée.

---

## Contact

Pour toute question ou suggestion, contactez-moi à l'adresse suivante :
**antonin.larvor.25@seatech.fr**.

N'hésitez pas à me faire part de vos retours ou des modifications que vous souhaitez apporter à ce README !

