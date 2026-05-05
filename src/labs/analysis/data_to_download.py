import csv
import requests
import os
import json

with open("data/MetObjects.csv", newline='', encoding='utf-8-sig') as museum_csv:
    museum_info = csv.DictReader(museum_csv)
    object_info = [row for row in museum_info if row['Classification'] == 'Paintings'][1234]

object_id = object_info['Object ID']

response = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}")
object_data = response.json()

image = object_data['primaryImage']
image_response = requests.get(image)

os.makedirs("paintings", exist_ok=True)
image_path = os.path.join("paintings", "1972_278_8_O.jpg")
with open(image_path, "wb") as f:
    f.write(image_response.content)

json_path = os.path.join("paintings", "1972_278_8_O.json")
with open(json_path, "w") as f:
    json.dump(object_data, f, indent=2)