from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Load the data
def create_embeddings(file):
    loader = CSVLoader(file, encoding='utf-8')
    data = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(data, embeddings)
    store = db.save_local(r"faiss_store")
    return store


def main():
    create_embeddings("") #Enter your file name
    print(" Created Embeddings! ")
    

if __name__ == "__main__":
    main()
