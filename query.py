from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def generate_response(embedding_path):
    embeddings = HuggingFaceEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, api_key="") #Enter your API Key
    vector = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
    db = vector.as_retriever()
    template = """You are a bot Specially designed to help the user to get the answers from the indexed CSV documents based on the user query.
                    You are intended to understand the user query and provide the answer from the indexed documents.You are strictly prohibited to provide the answer from the external sources.

    <context>
    {context}
    </context>

    Question: {input}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    # Create a chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(db, doc_chain)
    return chain


def main():
    chain = generate_response("faiss_store")
    #Input your Query here
    user_input = "Compare last 3 year vendor wise procurement for SAP department" 
    response = chain.invoke({"input": user_input})
    # print(response)
    print(response['answer'].replace("**", ""))
    sources = []
    if 'context' in response:
        sources = [doc.metadata['source'] for doc in response['context']]
        print("Sources: ", sources[0])

if __name__ == "__main__":
    main()
