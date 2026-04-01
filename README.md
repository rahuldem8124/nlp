# NLP Chatbot using Transformers

## Objective
This project implements a conversational AI chatbot using the **Hugging Face Transformers** library. The chatbot uses a pre-trained **DialoGPT-medium** model by Microsoft to generate context-aware, natural language responses.

## Features
- **Model**: [microsoft/DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium)
- **Framework**: PyTorch & Transformers
- **Interaction**: Multi-turn conversational loop with history management.
- **Improved Logic**: Custom generation parameters (`temperature=0.6`, `repetition_penalty=1.2`) to ensure logical and coherent answers.
- **Exit Condition**: Supports 'exit' or 'quit' to terminate the session.

## Files
- `chatbot.ipynb`: The main Jupyter Notebook containing the full implementation and walkthrough.
- `chatbot_interactive.py`: A terminal-ready version of the chatbot for quick testing.
- `requirements.txt`: List of necessary Python libraries.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `chatbot.ipynb` in Jupyter Notebook or Google Colab and run all cells.
3. Alternatively, run the terminal version:
   ```bash
   python chatbot_interactive.py
   ```

## Author
[Your Name/Username]
Data Science Intern - February 2026 Task
