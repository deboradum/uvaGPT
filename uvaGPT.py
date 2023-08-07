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
import textract
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class UvaGPT:
    def __init__(self, model="all"):
        load_dotenv()
        self.pdf_file = f"PDFs/{model}.pdf"
        self.txt_file = f"txts/{model}.txt"
        self.db_name = f"DBs/{model}.pkl"

        if not os.path.isfile(self.txt_file):
            print(f"Creating {self.txt_file}...")
            self.parse_pdf()
        if not os.path.isfile(self.db_name):
            print(f"Splitting {self.txt_file}...")
            chunks = self.split_txt()
            print(f"Creating {self.db_name}...")
            self.create_faiss_db(chunks)

        self.load_faiss_db()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.05), self.db.as_retriever(), memory=memory)


    def run(self):
        print("Starting chat. Type 'q' or 'exit' to quit.")
        while True:
            query = input("Ask your question: ")
            if not query:
                continue
            if query == "q" or query == "exit":
                return
            res = self.search(query)
            print(res)
            query = ""


    def parse_pdf(self):
        text = textract.process(self.pdf_file)
        with open(self.txt_file, 'w+') as f:
            f.write(text.decode('utf-8'))


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
        faiss.write_index(store.index, "chunks.index")
        store.index = None
        with open(self.db_name, "wb") as f:
            pickle.dump(store, f)


    def load_faiss_db(self):
        index = faiss.read_index("chunks.index")

        with open(self.db_name, "rb") as f:
            db = pickle.load(f)

        db.index = index
        self.db = db


    def search(self, query):
        result = self.qa_chain({"question": query})
        return result["answer"]


all_models = ['all', 'arco']

parser = argparse.ArgumentParser()
# Automatically create 'all' database.
parser.add_argument('-m', '--model', type=str, default="all",
                    help=f'The model (course) you want to query. \
                    Model must be one of these values: {all_models}, and defaults \
                    to all models.')

model = parser.parse_args().model

if model not in all_models:
    print(f"Model: {model} not supported. please pick one of the following models: {all_models}")
    exit()

uva = UvaGPT()
uva.run()
