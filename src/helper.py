from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load() # here we are running the loader that return a list of Document objects each with page_content and metadata
    return documents
    
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given a list of Document objects, return a new list of Document objects
       containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src} if src else {}
            )
        )
    return minimal_docs

# split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks
    
    
def download_embeddings():
    """Download and return the Huggingface embedding model.
    """
    # 1. Load a pretrained Sentence Transformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings