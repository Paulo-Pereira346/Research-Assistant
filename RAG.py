from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

vector_store = Chroma(
    persist_directory = './chroma_db/',
    embedding_function = embeddings,
    collection_name = 'notes'
)


def index_notes():
    notes_path = Path("notes")
    if not notes_path.exists():
        raise FileNotFoundError("The notes directory does not exist.")

    text_loader = DirectoryLoader(
        str(notes_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    pdf_loader = DirectoryLoader(
        str(notes_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = text_loader.load() + pdf_loader.load()
    if not documents:
        raise FileNotFoundError("No .txt or .pdf files were found in the notes directory.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)
    ids = [
        hashlib.sha256(
            f"{chunk.metadata.get('source', '')}:{index}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        for index, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(chunks, ids=ids)
    return len(chunks)

retriever = vector_store.as_retriever(search_kwargs = {"k" : 3})

llm = ChatGroq(
    model_name = os.environ['GROQ_MODEL']
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't there, say "I don't have that in my notes."

Context: {context}
Question: {input}
""")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask_langchain(question):
    answer = chain.invoke(question)
    return answer

if __name__ == "__main__":
    indexed_chunks = index_notes()
    print(f"Indexed {indexed_chunks} note chunks.")
    answer = ask_langchain("What is Supervised Learning? What are the two types?")
    print(answer)