import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from FlagEmbedding import FlagModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import pickle

processor = BlipProcessor.from_pretrained("/home/wyj/blip-image-captioning-base")
model_image_text = BlipForConditionalGeneration.from_pretrained("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/saved_blip")

model_text = FlagModel('/home/wyj/bge-base-en-v1.5',
                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                       use_fp16=True)

with open("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/sorted_Ours_mplug_owl3_refined_desc.txt", "r") as file:
    text_content = file.readlines()

def split_document_by_lines(text_lines):
    return [line.rstrip('\n') for line in text_lines]  # Remove empty lines


docs = split_document_by_lines(text_content)

def get_image_files_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  
    return [os.path.join(folder_path, f) for f in files]

image_folder = "/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/images_total"
image_files = get_image_files_from_folder(image_folder)

if len(image_files) != len(docs):
    raise ValueError("Not matched")

def generate_cross_modal_embeddings(image_files, texts, model, processor):
    embeddings = []
    for image_file, text in zip(image_files, texts):
        image = Image.open(image_file)
        inputs = processor(images=image, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)  
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

        embedding = np.expand_dims(embedding, axis=0)
        embeddings.append(embedding)
    return embeddings

def generate_text_embeddings(chunks, model):
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    return embeddings

cross_modal_embeddings = generate_cross_modal_embeddings(image_files, docs, model_image_text, processor)
text_embeddings = generate_text_embeddings(docs, model_text)

embedding_dict = {tuple(cross_modal_embeddings[i].flatten()): text_embeddings[i] for i in range(len(cross_modal_embeddings))}

with open('Cross_Modal_Database.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)

with open('Text_Database.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("The database is successfully generated.")

