# RA-ADC

## Project Overview
This is a project about **Retrieval-Augmented Autonomous Driving Corner Case Comprehension with Vision-Language Models**, aiming to **mitigate the hallucination phenomenon in VLMs through retrieval-augmented generation**. This project includes the following main features:

- **Feature 1**: Embedding of image-text pairs into vector database.
- **Feature 2**: Retrieving the most similar image with cosine similarity.
- **Feature 3**: Evaluating generated text with cosine similarity and ROUGE scores.

## Project Structure

```
Project Root Directory
├── embedding_&_retrieving/                # Source code of embedding and retrieving
├── fine-tuning/               # Source code of fine-tuning BLIP model
├── evaluating/              # Source code of evaluation
├── requirements.txt    # Dependencies
└── README.md           # Project description file
```

## Installation and Usage

### Prerequisites

- Python >= 3.9
- Other dependencies (list any required tools or libraries)

### Installation Steps

1. Clone the repository to your local machine:

```bash
[git clone https://github.com/your-username/project-name.git](https://github.com/yjwangtj/RA-ADC.git)
```

2. Navigate to the project directory and install dependencies:

```bash
cd /RA-ADC
pip install -r requirements.txt
```
### Notations

1. The original dataset of images and the corresponding text descriptions should be replaced with yours.

2. The codes of using different VLMs to generate answers are not included in this repository, and the conda environment should be solely reconstructed.
