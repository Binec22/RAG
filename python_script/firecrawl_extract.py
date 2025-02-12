from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
import json
from tqdm import tqdm  # Barre de progression

# Initialize the FirecrawlApp with your API key
app = FirecrawlApp(api_key='fc-f29ef3831a5948c18bd1183bf9e1dae1')


class ExtractSchema(BaseModel):
    data: str = Field(default="")
    key_words: str = Field(default="")


# Liste des URL à parcourir
# urls = [
#     'https://seatech.univ-tln.fr/-Devenir-Ingenieur-.html']
urls = [
    'https://seatech.univ-tln.fr/-Devenir-Ingenieur-.html',
    'https://seatech.univ-tln.fr/Parcours-Genie-maritime.html',
    'https://seatech.univ-tln.fr/Parcours-IngenieRie-et-sciences.html',
    'https://seatech.univ-tln.fr/Parcours-Innovation-Mecanique-pour-des-Systemes-Durables.html',
    'https://seatech.univ-tln.fr/Parcours-Materiaux-Durabilite-et.html',
    'https://seatech.univ-tln.fr/Parcours-Modelisation-et-Calculs.html',
    'https://seatech.univ-tln.fr/Parcours-Systemes-mecatroniques-et.html',
    'https://seatech.univ-tln.fr/Admission-sur-Concours.html',
    'https://seatech.univ-tln.fr/Admission-sur-titre.html',
    'https://seatech.univ-tln.fr/La-Prepa-des-INP.html',
    'https://seatech.univ-tln.fr/Formation-d-ingenieurs-Materiaux-par-apprentissage.html',
    'https://seatech.univ-tln.fr/Formation-d-ingenieurs-en-systemes-numeriques-par-apprentissage.html',
    'https://seatech.univ-tln.fr/Deroulement-des-etudes.html',
    'https://seatech.univ-tln.fr/-L-ecole-.html',
    'https://seatech.univ-tln.fr/Seatech-presentation-de-l-ecole-d-ingenieurs.html',
    'https://seatech.univ-tln.fr/Cadre-des-etudes.html',
    'https://seatech.univ-tln.fr/L-accompagnement-du-handicap-a-SeaTech.html',
    'https://seatech.univ-tln.fr/La-vie-etudiante-a-Seatech.html',
    'https://seatech.univ-tln.fr/L-association-des-anciens-eleves-de-SeaTech.html',
    'https://seatech.univ-tln.fr/-International-.html',
    'https://seatech.univ-tln.fr/Programmes-d-echanges.html',
    'https://seatech.univ-tln.fr/DOUBLES-DIPLOMES-ET-ECHANGES-INTERNATIONAUX.html',
    'https://seatech.univ-tln.fr/Stages-a-l-etranger-261.html',
    'https://seatech.univ-tln.fr/Enseignement-des-langues.html',
    'https://seatech.univ-tln.fr/L-insertion-professionnelle.html',
    'https://seatech.univ-tln.fr/Un-lien-fort-avec-le-contexte-industriel-regional.html',
    'https://seatech.univ-tln.fr/Une-formation-professionnalisante.html',
    'https://seatech.univ-tln.fr/Des-formations-liees-aux.html',
    'https://seatech.univ-tln.fr/Du-laboratoire-a-l-entreprise.html',
    'https://seatech.univ-tln.fr/Des-eleves-de-SeaTech-participent-pour-la-premiere-annee-au-Dassault-UAV.html',
]

# Chemin vers le fichier JSON où sauvegarder les données
output_file = "firecrawl_database4.json"

# Charger la base de données existante ou initialiser une nouvelle liste
try:
    with open(output_file, "r", encoding="utf-8") as file:
        database = json.load(file)
except FileNotFoundError:
    database = []  # Initialiser une base de données vide si le fichier n'existe pas

# Parcourir chaque URL avec une barre de progression et ajouter les données extraites
for url in tqdm(urls, desc="Extraction des données", unit="URL"):
    try:
        data = app.extract([url], {
            'prompt': "Extrayez les informations pertinentes du site web de l'école d'ingénieurs Seatech pour les transformer en données prêtes à être utilisées par un modèle de langage (LLM). Organisez les informations extraites sous deux catégories : 1. **data** : Contenu détaillé et structuré extrait de la page. 2. **key_word** : Liste de mots-clés résumant les thèmes principaux abordés. Assurez-vous que les données soient complètes, bien organisées et directement exploitables pour des tâches de traitement du langage.",
            'schema': ExtractSchema.model_json_schema(),
        })

        # Ajouter l'URL visitée aux données extraites
        data_with_url = {
            "url": url,
            **data  # Fusionner l'URL avec le contenu extrait
        }

        # Ajouter les données enrichies à la base de données
        database.append(data_with_url)
    except:
        pass

# Sauvegarder la base de données mise à jour dans le fichier JSON
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(database, file, ensure_ascii=False, indent=4)

print(f"Données extraites et sauvegardées dans {output_file}.")
