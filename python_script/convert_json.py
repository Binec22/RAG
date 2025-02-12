import json

# Chargement du fichier JSON depuis le disque
with open('firecrawl_database4.json', 'r', encoding='utf-8') as infile:
    input_json = json.load(infile)

# Transformation des données pour chaque élément
output_json = [
    {
        "data": {
            "text_content": item["data"]["data"],
            "key_words": item["data"]["key_words"],
            "url": item["url"]
        }
    }
    for item in input_json
]

# Écriture du résultat dans un nouveau fichier JSON
with open('firecrawl_database5.json', 'w', encoding='utf-8') as outfile:
    json.dump(output_json, outfile, indent=4, ensure_ascii=False)

print("Transformation terminée et sauvegardée dans 'output_file.json'")