import os
import pickle
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import json

# 加载图像编码模型
model_image = BlipForConditionalGeneration.from_pretrained("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/saved_blip")
processor_image = BlipProcessor.from_pretrained("/home/wyj/blip-image-captioning-base")

# 加载文本数据库
with open("/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mplug_owl3/sorted_Ours_mplug_owl3_refined_desc.txt", "r") as file:
    text_content = file.readlines()

def split_document_by_lines(text_lines):
    return [line.strip() for line in text_lines if line.strip()]

docs = split_document_by_lines(text_content)

# 加载嵌入和数据库
with open('Text_Database.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('Cross_Modal_Database.pkl', 'rb') as f:
    embedding_dict = pickle.load(f)

# 余弦相似度计算
def cosine_similarity(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0
    return np.dot(vec1, vec2) / denom

# 匹配嵌入索引
def find_matching_embedding(embeddings, target_embedding):
    for idx, embedding in enumerate(embeddings):
        if np.allclose(embedding, target_embedding, atol=1e-5):
            return idx
    return -1

# 遍历文件夹中的所有图像
image_folder_path = "/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/Database_mlpug_owl3/test/once"  # 替换为实际的图像文件夹路径
result_file_path = "once_retrieval_results.json"  # 输出的 JSON 文件路径
result_list = []

alpha = 0.1  # 权重调整

for image_file in os.listdir(image_folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 支持的图像格式
        image_path = os.path.join(image_folder_path, image_file)

        # 读取图像并生成嵌入
        new_image = Image.open(image_path)
        new_inputs = processor_image(images=new_image, text="", return_tensors="pt")
        with torch.no_grad():
            new_img_emb = model_image(**new_inputs, output_hidden_states=True)
        new_img_emb = new_img_emb.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

        # 检索数据库
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

        # 查找匹配的文本
        matching_idx = find_matching_embedding(embeddings, retrieved_text)
        if matching_idx != -1:
            retrieved_text_plain = docs[matching_idx]
            result = {
                "image_path": image_path,
                "retrieved_image_idx": retrieved_image_idx,
                "retrieved_text": retrieved_text_plain
            }
        else:
            result = {
                "image_path": image_path,
                "retrieved_image_idx": None,
                "retrieved_text": "Nothing is retrieved."
            }

        # 添加到结果列表
        result_list.append(result)

# 将结果写入 JSON 文件
with open(result_file_path, 'w') as json_file:
    for result in result_list:
        json_file.write(json.dumps(result) + "\n")

print(f"Results have been written to {result_file_path}")