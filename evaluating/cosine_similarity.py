import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from FlagEmbedding import FlagModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 加载文本编码模型（BGE）
model_text = FlagModel('/home/wyj/bge-base-en-v1.5',
                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                       use_fp16=True)

# 读取两个文本文件
with open("../answers/sorted_annotation.txt", "r") as file1:
    text1_lines = file1.readlines()

with open("../answers/internvl/without_rag_internvl-40b.txt", "r") as file2:
    text2_lines = file2.readlines()

# 确保两个文件行数一致
assert len(text1_lines) == len(text2_lines)

def split_document_by_lines(text_lines):
    chunks = []
    for line in text_lines:
        if line.strip():
            chunks.append(line.strip())
    return chunks

docs1 = split_document_by_lines(text1_lines)
docs2 = split_document_by_lines(text2_lines)

# 计算每一对文本行之间的余弦相似度
similarities = []

for line1, line2 in zip(docs1, docs2):
    # 获取每行文本的嵌入向量
    emb1 = model_text.encode(line1)  # 获取文本1的嵌入
    emb2 = model_text.encode(line2)  # 获取文本2的嵌入

    # 计算余弦相似度
    sim = cosine_similarity([emb1], [emb2])[0][0]
    similarities.append(sim)

# 计算平均余弦相似度
average_similarity = np.mean(similarities)
print(f"Average Cosine Similarity: {average_similarity}")