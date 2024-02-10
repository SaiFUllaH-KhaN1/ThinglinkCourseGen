from flask import Flask, render_template, request, Response, jsonify, session, send_from_directory, flash, redirect, url_for
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os
import openai
from langchain.chains.conversation.memory import ConversationBufferMemory
from openai import OpenAI
from langchain.prompts import BaseChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import requests
import json
import langchaindemo as LCD
import threading
import queue
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv(dotenv_path="E:\downloads\THINGLINK\dante\HUGGINGFACEHUB_API_TOKEN.env")
# Set the API key for OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import io
import os
os.environ["OPENAI_API_KEY"] = "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"


app = Flask(__name__)
app.secret_key = '123'

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

def llm_thread(g, prompt, last_messages):
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True, verbose= True,callbacks=[ChainStreamHandler(g)])
        embeddings = OpenAIEmbeddings()
        load_docsearch = FAISS.load_local("faiss_index",embeddings)
        chain, docs, query, subject_name = LCD.TALK_WITH_RAG(prompt, load_docsearch,llm)
        chating_history = last_messages
        print(docs)
        print("chating_history",chating_history)
        chain({"input_documents": docs, "human_input": query, "subject_name": subject_name,"chat_history": chating_history})
    finally:
        g.close()


def chain(prompt, last_messages):
    if prompt != "":
        g = ThreadedGenerator()
        threading.Thread(target=llm_thread, args=(g, prompt, last_messages)).start()
        return g

@app.route("/", methods=["GET", "POST"])
def index():
    f = None
    filename = None
    if 'file' in request.files:
        f = request.files.get('file')

    if f is not None:
        if not f.filename:
            flash("No file uploaded! Please upload file first.")
        else:
            filename = f.filename
            # f_saved_path = 'E:\\downloads\\THINGLINK\\dante'  # Change this to the directory where you want to save the files
            # f_saved = os.path.join(f_saved_path, f.filename)
            # f.save(f_saved)
            file_content = io.BytesIO(f.read())
            # Now, you can use file_content as a file-like object
            docsearch = LCD.RAG(file_content) 
            # Now, open the saved file using PyPDF2
            # docsearch = LCD.RAG(open(f_saved, 'rb'))  

            docsearch.save_local("faiss_index")
    return render_template('index.html', filename=filename)

@app.route("/get", methods=["GET", "POST"])
def chat():
    data = request.json
    user_input = data.get('prompt')
    chating_history = data.get('chating_history')
    last_messages = chating_history[-5:]
    print("Last messages at the /get",last_messages)
    return Response(chain(user_input, last_messages),mimetype='text/plain') # return get_Chat_response(input)   

# @app.route("/chating_history", methods=["GET", "POST"])
# def chating_history():
#     chating_history = request.json['chating_history']
#     last_10_messages = chating_history[-10:]
#     return last_10_messages

@app.route('/graphml', methods=['GET', 'POST'])
def graphml():
    graphml_content = ""
    plot_image_uri = ""
    output_graphml = ""
    path_graphml = 'E:\downloads\THINGLINK\dante\graph.graphml'
    if request.method == 'GET':
        text_data = request.args.get('textData')
        if text_data:
            output_graphml = LCD.GENERATE_GRAPHML(text_data)
            graphml_file_path = 'E:\downloads\THINGLINK\dante'
            with open(os.path.join(graphml_file_path, 'graph.graphml'), 'w', encoding='utf-8') as graphml_file:
                graphml_file.write(output_graphml)
            
            with open(path_graphml, 'r') as graphml_file:
                graphml_content = graphml_file.read()
    else:  # This block handles POST requests
        with open(path_graphml, 'r') as graphml_file:
            graphml_content = graphml_file.read()
        width = request.form.get('width')
        height = request.form.get('height')
        if width and height:
            try:
                plot_image_uri = LCD.DRAW_GRAPH(path_graphml, width, height)
                graphml_content = graphml_content
                flash("The GraphML generated was compilable!")
            except:
                flash("The GraphML is not compilable, Go back to Regenerate! ")

    return render_template("index_copy.html", debug_output_graphml=graphml_content, img_uri=plot_image_uri, text_data=output_graphml)


if __name__ == '__main__':
    app.run()
