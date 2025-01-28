import json
import openai
from langchain_openai import ChatOpenAI
import os

openai.api_key = "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"


def reformulate_text(text):
    """
    Reformule le texte en utilisant le modèle OpenAI chargé via LangChain.
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai.api_key)
    """
    Reformule le texte en utilisant le modèle OpenAI chargé via LangChain.
    """
    try:
        prompt = (
            "Reformule ce texte pour le rendre exploitable dans une base de données destinée à un système de RAG d'une école d'ingénieur. "
            "Le texte doit :\n"
            "1. Être concis et explicite.\n"
            "2. Mettre en avant les informations clés et les thèmes abordés.\n"
            "3. Être rédigé de manière à répondre potentiellement à des questions d’utilisateurs.\n\n"
            f"Texte original :\n{text}\n\nTexte reformulé :"
        )
        response = model.generate([prompt])

        return response.generations[0][0].text.strip()

    except Exception as e:
        print(f"Erreur lors de la reformulation avec LangChain : {e}")
        return text  # Retourne le texte original en cas d'erreur


def process_json_file(input_file, output_file):
    """
    Lit un fichier JSON, reformule le champ "markdown", et enregistre un nouveau fichier JSON.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        counter = 0
        for entry in data:
            if counter == 0:
                print(entry)
            if "data" in entry:
                counter += 1
                if counter == 1:
                    print("\n\n")
                # print(f"Reformulation du texte {counter} : {entry['markdown'][:50]}...")  # Affiche un aperçu
                # entry["markdown"] = reformulate_text(entry["markdown"])

        # with open(output_file, "w", encoding="utf-8") as file:
        #     json.dump(data, file, ensure_ascii=False, indent=2)

        print(f"Traitement terminé. Résultat enregistré dans {counter}")

    except Exception as e:
        print(f"Erreur lors du traitement du fichier JSON : {e}")


# Chemins des fichiers
#input_json = "python_script/un_texte.json"
input_json = "firecrawl_database4.json"
output_json = "python_script/base_de_donnees_reformulee2.json"

# Appel de la fonction principale
process_json_file(input_json, output_json)
