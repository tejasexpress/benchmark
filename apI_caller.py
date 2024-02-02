import json
import os
import shutil
import requests
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import time


PROJECT_ID = "ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


# Read JSON data from file
with open('TextVQA_0.5.1_val.json', 'r') as file:
    json_data = file.read()

# Parse JSON data
parsed_data = json.loads(json_data)

# Extract image_ids into a list
vqas = [item for item in parsed_data["data"]]
first_200_vqa = vqas[300:500]

def api_caller(vqas, source_folder, destination_folder):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-pro-vision")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_folder_path = os.path.join(script_dir, source_folder)
    result = []
    count = 0
    for vqa in vqas:
        image_id = vqa["image_id"]
        question_text = "Provide concise and direct responses to the following questions in very few words. Avoid using complete sentences and refrain from adding any additional information or context. Answer objectively. if you think the question is unanswerable then say unanswerable : "+ vqa["question"]
        image_filename = f"{image_id}.jpg"
        source_path = os.path.join(source_folder_path, image_filename)
        source_image = Image.load_from_file(location=source_path)
        time.sleep(3)
        try:
            answers = model.generate_content([question_text, source_image])
            count+=1
            print(answers.text + f" : {count} / 200")
            question_id = vqa["question_id"]
            print(image_filename)
            result.append({"question_id": question_id, "answer": answers.text})
        except Exception as e:
            print("rate limited")           

    return result


        

# Example usage
source_folder_relative_path = "val_folder"
destination_folder_path = "answers"

result = api_caller(first_200_vqa, source_folder_relative_path, destination_folder_path)
with open('result3.json', 'w') as file:
    json.dump(result, file)
