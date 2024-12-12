import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from FlagEmbedding import FlagModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import pickle

# Loading Xmodal processor and model
processor = BlipProcessor.from_pretrained("/home/wyj/blip-image-captioning-base")
model_image_text = BlipForConditionalGeneration.from_pretrained("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/saved_blip")

# Loading text encoding model
model_text = FlagModel('/home/wyj/bge-base-en-v1.5',
                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                       use_fp16=True)

# Load TXT file
with open("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/sorted_Ours_mplug_owl3_refined_desc.txt", "r") as file:
    text_content = file.readlines()

# Chunking by lines
def split_document_by_lines(text_lines):
    return [line.rstrip('\n') for line in text_lines]  # Remove empty lines


docs = split_document_by_lines(text_content)

def get_image_files_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # 按文件名排序
    return [os.path.join(folder_path, f) for f in files]

image_folder = "/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/images_total"
image_files = get_image_files_from_folder(image_folder)

if len(image_files) != len(docs):
    raise ValueError("图像文件数量与文本块数量不一致，无法建立一一对应关系。")

# Generating XModal vectors for images and texts
def generate_cross_modal_embeddings(image_files, texts, model, processor):
    embeddings = []
    for image_file, text in zip(image_files, texts):
        image = Image.open(image_file)
        inputs = processor(images=image, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)  # Obtaining the hidden states of output
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

# Creating a dictionary matching XModal embeddings and text embeddings
embedding_dict = {tuple(cross_modal_embeddings[i].flatten()): text_embeddings[i] for i in range(len(cross_modal_embeddings))}

'''
# 创建索引，将跨模态嵌入向量添加到索引中
dimension = cross_modal_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
embedding_array = np.array(cross_modal_embeddings)
index.add(embedding_array)

# 保存索引和嵌入到文件
faiss.write_index(index, 'combined_faiss_vector_database.index')
'''

with open('Cross_Modal_Database.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)

with open('Text_Database.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("The database is successfully generated.")

