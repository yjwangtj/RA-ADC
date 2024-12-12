import pickle
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from FlagEmbedding import FlagModel


model_image = BlipForConditionalGeneration.from_pretrained("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/saved_blip")
processor_image = BlipProcessor.from_pretrained("/home/wyj/blip-image-captioning-base")




with open("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/sorted_Ours_mplug_owl3_refined_desc.txt", "r") as file:
    text_content = file.readlines()


def split_document_by_lines(text_lines):
    chunks = []
    for line in text_lines:
        if line.strip():
            chunks.append(line.strip())
    return chunks

docs = split_document_by_lines(text_content)

with open('Text_Database.pkl', 'rb') as f:
    embeddings = pickle.load(f)


with open('Cross_Modal_Database.pkl', 'rb') as f:
    embedding_dict = pickle.load(f)


new_image = Image.open("/home/wyj/1/base-val-1500/images/000001_1616005007200.jpg")
new_text = ""
new_inputs = processor_image(images=new_image, text=new_text, return_tensors="pt")
with torch.no_grad():
    new_img_emb = model_image(**new_inputs, output_hidden_states=True)
new_img_emb = new_img_emb.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

def cosine_similarity(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0
    return np.dot(vec1, vec2) / denom


alpha = 0 
max_similarity = -1
retrieved_text = None
retrieved_image_idx = None

for idx, img_emb in enumerate(embedding_dict.keys()):
    img_emb_array = np.array(img_emb)
    img_similarity = cosine_similarity(new_img_emb, img_emb_array[:len(new_img_emb)])
    text_similarity = cosine_similarity(new_img_emb, img_emb_array[len(new_img_emb):])
    similarity = (1 - alpha) * img_similarity + alpha * text_similarity

    if similarity > max_similarity:
        max_similarity = similarity
        retrieved_text = embedding_dict[img_emb]
        retrieved_image_idx = idx + 1  

print("Max similarity:", max_similarity)

def find_matching_embedding(embeddings, target_embedding):
    for idx, embedding in enumerate(embeddings):
        if np.allclose(embedding, target_embedding, atol=1e-5):
            return idx
    return -1


matching_idx = find_matching_embedding(embeddings, retrieved_text)
if matching_idx != -1:
    retrieved_text_plain = docs[matching_idx]
    print("Successfully retrieved text:")
    print(retrieved_text_plain)
    print("Matching image index:", retrieved_image_idx)  # Print the image index
else:
    print("Nothing is retrieved.")
