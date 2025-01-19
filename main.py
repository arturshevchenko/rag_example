import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


def get_llm() -> ChatOpenAI:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the language model
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0.7
    )


def ask_question_without_context(question: str) -> str:
    # Initialize the language model
    llm = get_llm()

    answer = llm.invoke(question)
    return answer.content


def ask_question_with_context(question: str, documents: list[str]) -> str:
    # Initialize the language model
    llm = get_llm()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=200  # Overlap between chunks
    )

    # Split documents into chunks
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc)
        chunks.extend(splits)

    # Create FAISS vector store from chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    # Create a FAISS vector store from the documents
    # vector_store = FAISS.from_texts(documents, embeddings)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    context_aware_question = (
        f"Use FIRSTLY the provided context as the primary source of information to answer the following question even if it conflicts with common knowledge. "
        f"If the answer is not in the context, start finding in you base of knowledge. \n\n"
        f"{question}"
    )
    retrieved_context = vector_store.as_retriever().get_relevant_documents(question)
    print(f"Context: {retrieved_context}")
    answer = qa_chain.invoke({"query": context_aware_question})
    return answer.get('result')


def load_txt_files(directory: str) -> list[str]:
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
    return texts


def example_1():
    documents = load_txt_files("docs")

    # Example questions
    questions = [
        "What is the capital of France?",
        "Who created Python?",
        "What is RRT?",
        # "What is PPL?",
        # "What i get if combine RRT and PPL?",
        # "What game i can play in monopoly?"
    ]

    for question in questions:
        answer = ask_question_with_context(question, documents)
        print(f"Answer with context\nQ: {question}\nGPT-Answer: {answer}\n")
        answer_simple = ask_question_without_context(question)
        print(f"Simple answer\nQ: {question}\nGPT-Answer: {answer_simple}\n{'-' * 10}")


def example_2():
    documents = load_txt_files("docs")

    # Example questions
    questions = [
        "Generate hight level test scenaios for log in feature? Consider positive and negative scenarios",
    ]

    for question in questions:
        answer = ask_question_with_context(question, documents)
        print(f"Answer with context\nQ: {question}\nGPT-CONTEXT-Answer: {answer}\n")
        answer_simple = ask_question_without_context(question)
        print(f"Simple answer\nQ: {question}\nGPT-Answer: {answer_simple}\n{'-' * 10}")


if __name__ == "__main__":
    example_1()
