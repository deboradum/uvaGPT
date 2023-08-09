import argparse
from dotenv import load_dotenv
import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import pickle
from pdfminer.high_level import extract_text
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def get_all_models():
    return [m.removesuffix(".pdf") for m in os. listdir("PDFs/") if m != ".DS_Store"]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        type=str,
                        default="arco", # terug naar all doen wanneer geimplementeerd
                        choices= ALL_MODELS,
                        help=f'The model (course) you want to query. \
                        Model must be one of these values: {ALL_MODELS}, and defaults \
                        to all models.')

    model = parser.parse_args().model
    if model not in ALL_MODELS:
        print(f"Model: {model} not supported. please pick one of the following models: {ALL_MODELS}")
        exit()

    return model


def combine_dbs():

    return


class UvaGPT:
    def __init__(self, model="all"):
        load_dotenv()
        self.model = model
        self.pdf_file = f"PDFs/{model}.pdf"
        self.txt_file = f"txts/{model}.txt"
        self.db_name = f"DBs/{model}.pkl"
        self.chunks_path = f"chunks/{model}_chunks.index"

        # TODO: Automatically create 'all' database.
        if not os.path.isfile(self.txt_file) and self.model != "all":
            print(f"Creating {self.txt_file}...")
            self.parse_pdf()
        if not os.path.isfile(self.db_name) and self.model != "all":
            print(f"Splitting {self.txt_file}...")
            chunks = self.split_txt()
            print(f"Creating {self.db_name}...")
            self.create_faiss_db(chunks)

        if self.model == "all" and not os.path.isfile(self.db_name):
            combine_dbs()

        self.load_faiss_db()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Maybe 3.5 turbo gebruiken ipv davinci?
        self.qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.05), self.db.as_retriever(), memory=memory)


    def run(self):
        print("Starting chat. Type 'q' or 'exit' to quit.")
        while True:
            query = input(f"Model: {model}GPT. Ask your question: ")
            if not query:
                continue
            if query == "q" or query == "exit":
                return
            res = self.search(query)
            print(res)
            query = ""


    def parse_pdf(self):
        text = extract_text(self.pdf_file)
        with open(self.txt_file, 'w+') as f:
            f.write(text)


    def split_txt(self):
        with open(self.txt_file, "r") as f:
            text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=1250, separator="\n")
        chunks = []
        splits = text_splitter.split_text(text)
        chunks.extend(splits)

        return chunks


    def create_faiss_db(self, chunks):
        store = FAISS.from_texts(chunks, OpenAIEmbeddings())
        faiss.write_index(store.index, self.chunks_path)
        store.index = None
        with open(self.db_name, "wb") as f:
            pickle.dump(store, f)


    def load_faiss_db(self):
        index = faiss.read_index(self.chunks_path)

        with open(self.db_name, "rb") as f:
            db = pickle.load(f)

        db.index = index
        self.db = db


    def search(self, query):
        result = self.qa_chain({"question": query})
        return result["answer"]


ALL_MODELS = get_all_models()
model = parse()
uva = UvaGPT(model=model)
uva.run()
