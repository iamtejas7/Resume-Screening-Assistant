from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain


#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# iterate over files in 
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        chunks=get_pdf_text(filename)
        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    #print(docs)
    return docs


#Create embeddings instance
def create_embeddings_load_data():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings


#Function to push data to Vector Store
def push_to_faiss(embeddings, docs):
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    


#Function to pull infrmation from Vector Store
def pull_from_faiss(embeddings, faiss_index_name):
    index = FAISS.load_local(faiss_index_name, embeddings, allow_dangerous_deserialization=True)
    return index


#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,faiss_index_name,embeddings,unique_id):
    index_name = faiss_index_name
    index = pull_from_faiss(embeddings, index_name)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary