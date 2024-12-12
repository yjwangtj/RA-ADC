import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from FlagEmbedding import FlagModel

# Load the BLIP model and text model
processor = BlipProcessor.from_pretrained("/home/wyj/blip-image-captioning-base")
model_image_text = BlipForConditionalGeneration.from_pretrained("/home/wyj/blip-image-captioning-base").to('cuda')
model_text = FlagModel('/home/wyj/bge-base-en-v1.5',
                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                       use_fp16=True)


# Dataset definition
class XModalAttentionDataset(Dataset):
    def __init__(self, image_path, text_chunks):
        self.image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if
                            os.path.isfile(os.path.join(image_path, f))]
        self.text_chunks = text_chunks
        if len(self.image_files) != len(self.text_chunks):
            raise ValueError("Number of images and text chunks do not match")

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((720, 1280)),  # Resize to consistent dimensions
            transforms.ToTensor(),  # Convert to tensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        text = self.text_chunks[idx]

        # Apply transformations
        image = self.transform(image)
        return image, text


# Contrastive Loss definition with temperature
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, similarity_matrix):
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        loss_img = F.cross_entropy(similarity_matrix / self.temperature, labels)
        loss_txt = F.cross_entropy(similarity_matrix.T / self.temperature, labels)
        return (loss_img + loss_txt) / 2


# Hard negative mining function
def hard_negative_mining(image_embeds, text_embeds, margin=0.2):
    similarity_matrix = cosine_similarity(image_embeds.detach().cpu().numpy(), text_embeds.detach().cpu().numpy())
    hard_negatives_img, hard_negatives_txt = [], []

    for i, sims in enumerate(similarity_matrix):
        sorted_indices = np.argsort(sims)[::-1]
        for idx in sorted_indices:
            if idx != i:
                hard_negatives_img.append(image_embeds[i])
                hard_negatives_txt.append(text_embeds[idx])
                break

    if len(hard_negatives_img) == 0 or len(hard_negatives_txt) == 0:
        # If no hard negatives found, return original embeddings
        return image_embeds, text_embeds

    return torch.stack(hard_negatives_img).to('cuda'), torch.stack(hard_negatives_txt).to('cuda')


# Training function
def train_model(image_path, text_chunks, model, processor, epochs=200, batch_size=16, lr=1e-5,
                use_semi_hard_negative=False):
    dataset = XModalAttentionDataset(image_path, text_chunks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for images, texts in dataloader:
            images = images.to('cuda')

            image_embeds, text_embeds = [], []
            for image, text in zip(images, texts):
                # Process image embedding
                inputs = processor(images=image, text=text, return_tensors="pt").to('cuda')
                outputs = model(**inputs, output_hidden_states=True)
                image_embed = outputs.hidden_states[-1].mean(dim=1).squeeze().float()
                image_embeds.append(image_embed)

                # Process text embedding with gradient
                text_embed = torch.tensor(model_text.encode(text), requires_grad=True).float().to('cuda')
                text_embeds.append(text_embed)

            image_embeds = torch.stack(image_embeds).to('cuda')
            text_embeds = torch.stack(text_embeds).to('cuda')

            # Hard negative mining
            hard_images, hard_texts = hard_negative_mining(image_embeds, text_embeds)
            similarity_matrix = torch.mm(F.normalize(hard_images, dim=-1), F.normalize(hard_texts, dim=-1).T)
            loss = criterion(similarity_matrix)

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
        scheduler.step()

    model.save_pretrained(
        "/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/saved_blip/")


# Load TXT file and split into chunks
with open(
        "/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mini-internvl/sorted_Ours_mini_internvl_refined_desc.txt",
        "r") as file:
    text_content = file.readlines()


def split_document_by_lines(text_lines):
    return [line.rstrip('\n') for line in text_lines]  # Remove empty lines


docs = split_document_by_lines(text_content)

image_path = '/home/wyj/0-Research_Projects/Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models/datasets/mini-internvl/images_total/'

# Start training
train_model(image_path, docs, model_image_text, processor, lr=1e-5)

