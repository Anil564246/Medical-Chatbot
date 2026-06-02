# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask

# Project Overview
This project demonstrates how to build a complete end-to-end Medical Chatbot powered by Generative AI. This chatbot is capable of understanding medical queries, retrieving relevant information, and generating intelligent, context-aware responses in real time.
It integrates Large Language Models (LLMs) with LangChain for orchestration, Pinecone for vector search, Flask for backend deployment, and AWS for scalable hosting.

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Anil564246/Medical-Chatbot.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone


### Demo Images
<img src="C:\Users\Anil Prajapati\Desktop\Projects\Medical-Chatbot\medical_chatbot.egg-info" alt="Demo" width="400">
<img src="C:\Users\Anil Prajapati\Desktop\Projects\Medical-Chatbot\Images\Screenshot 2026-06-02 120917.png" alt="Demo" width="400">
<img src="C:\Users\Anil Prajapati\Desktop\Projects\Medical-Chatbot\Images\Screenshot 2026-06-02 120929.png" alt="Demo" width="400">