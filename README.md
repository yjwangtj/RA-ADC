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
git clone https://github.com/your-username/project-name.git
```

2. Navigate to the project directory and install dependencies:

```bash
cd project-name
pip install -r requirements.txt
```

### Running the Project

Use the following command to start the project:

```bash
python src/main.py
```

## Usage

1. **Feature 1**: Explain how to use this feature.
2. **Feature 2**: Explain how to use this feature.
3. **Feature 3**: Explain how to use this feature.

## Testing

Run the following command to execute tests:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! If you have ideas or find issues, please submit an issue or pull request.

Before contributing, please read the [CONTRIBUTING.md](CONTRIBUTING.md) to understand our contribution guidelines.

## License

This project is open-sourced under the [MIT License](LICENSE). See the LICENSE file for details.

## Contact

If you have any questions or suggestions, feel free to contact us:

- Email: your_email@example.com
- GitHub Issues: [Project Link](https://github.com/your-username/project-name/issues)

