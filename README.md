# RAG simple example

## Overview
This project is a Python-based application designed for intelligent question-answering using OpenAI's advanced language models, such as `gpt-3.5-turbo`. It leverages LangChain for building sophisticated retrieval-based question-answering systems, allowing users to ask questions both with and without contextual documents.

## Features
- **Context-based Question-Answering**: Uses a vector database (e.g., FAISS) to retrieve relevant contextual information from documents to enhance the accuracy and relevance of responses.
- **Context-free Question-Answering**: Directly queries OpenAI's language model for general knowledge responses without document context.
- **Automated Document Processing**: Automates the process of reading and splitting text files into manageable chunks for optimal retrieval and embedding.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an `.env` file for environment-specific settings:
   - Add OPENAI_API_KEY variable in the `.env` file.

5. Run the application:
   Specify the command or script here, for example:
   ```bash
   python main.py
   ```

### Environment Setup

- Ensure that you have Python 3.x installed on your system.
- Follow the installation instructions above to set up your environment.

### Adding Dependencies

For adding extra dependencies, you can use:
```bash
pip install <package_name>
pip freeze > requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
