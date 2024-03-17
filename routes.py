from flask import Flask, render_template, request, Response, jsonify, session, send_from_directory, flash, redirect, url_for
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os
import openai
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
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

load_dotenv(dotenv_path="HUGGINGFACEHUB_API_TOKEN.env")
# Set the API key for OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import io
import os

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

def llm_thread(g, prompt, chating_history,scenario):
    try:

        # #Memory Make
        # memory = ConversationBufferMemory(return_messages=True)
        # # Iterate over each pair of user and bot messages
        # for pair in chating_history:
        #     user_message = pair['user']
        #     bot_message = pair['bot']
        #     # Save the context of each conversation pair to memory
        #     memory.save_context({"input": user_message}, {"output": bot_message})
        # llm_memory = memory.load_memory_variables({})

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1, streaming=True, verbose= True,callbacks=[ChainStreamHandler(g)])
        embeddings = OpenAIEmbeddings()
        load_docsearch = FAISS.load_local("faiss_index",embeddings)
        print("SCENARIO ====",scenario)
        chain, docs_main, query, subject_name = LCD.TALK_WITH_RAG(prompt, load_docsearch,llm,scenario,chating_history)
        # chating_history = last_messages
        # print(docs_main)
        # print("chating_history",chating_history)
        # chain.run({"human_input": query, "subject_name": subject_name, "input_documents": docs_main}) #,"chat_history": chating_history})
        # chain.invoke({"input_documents": docs_main,"subject_name": subject_name,"human_input": query,"chat_history": chating_history})
        chain({"input_documents": docs_main,"subject_name": subject_name,"human_input": query})
    finally:
        g.close()


def chain(prompt, chating_history,scenario):
    if prompt != "":
        g = ThreadedGenerator()
        threading.Thread(target=llm_thread, args=(g, prompt, chating_history,scenario)).start()
        return g

@app.route("/", methods=["GET", "POST"])
def index():

    if 'file' in request.files:
        f = request.files.get('file')
        print("There is a file")
        filename = f.filename
        file_content = io.BytesIO(f.read())
        docsearch = LCD.RAG(file_content)
        docsearch.save_local("faiss_index")
        session['file_uploaded'] = True
    else:
        f = None
        filename = None
        
    if f is not None:
        if not f.filename:
            flash("No file uploaded! Please upload file first.")
            session['file_uploaded'] = False
        else:
            session['file_uploaded'] = True

    return render_template('index.html', filename=filename, file_uploaded=session.get('file_uploaded', False))


@app.route("/get", methods=["GET", "POST"])
def chat():
    data = request.json
    user_input = data.get('prompt')
    chating_history = data.get('chating_history')
    scenario = data.get('scenarioState')
    if scenario:
        scenario = int(scenario)
    else:
        scenario = 0
    chating_history = chating_history[-10:]
    # print("Last messages at the /get",last_messages)
    # memory = memory
    # last_bot_message = chating_history[-1].get('bot', "") if chating_history else ""
    # memory.save_context({"input": user_input}, {"output": last_bot_message})
    # memory.load_memory_variables({})

    return Response(chain(user_input, chating_history, scenario),mimetype='text/plain') # return get_Chat_response(input)   


@app.route('/graphml', methods=['GET', 'POST'])
def graphml():
    text_datas = ""
    graphml_content = ""
    plot_image_uri = ""
    output_graphml = ""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_graphml = os.path.join(current_directory, 'graph.graphml')
    if request.method == 'POST':
        if request.is_json:  # If request contains JSON data
            data = request.json
            text_data = data.get('graphmltext')
            width = data.get('width')
            height = data.get('height')
            graphcode = data.get('graphcode')
            if text_data and not width and not height:
                print("if text_data and not width and not height")
                usr_inp = text_data
                # usr_inp = request.args.get('usr_inp', default='')
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
                        print(token)
                        self.gen.send(token)

                def llm_thread(g,prompt):
                    try:
                        llmsx = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True, callbacks=[ChainStreamHandler(g)],verbose=True)
                        graphml_chain = LCD.GENERATE_GRAPHML(prompt,llmsx)
                        graphml_chain.run(text=prompt)
                    finally:
                        g.close()

                def chain(prompt):
                    if prompt != "":
                        g = ThreadedGenerator()
                        threading.Thread(target=llm_thread, args=(g,prompt)).start()
                        return g
#                 graphmlml='''<?xml version="1.0" encoding="UTF-8"?>
# <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
#     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#     xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
#     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
# <key id="d0" for="edge" attr.name="description" attr.type="string"/>
# <graph id="G" edgedefault="undirected">
#     <!-- Nodes -->
#     <node id="Topic"/>
#     <node id="Scenario"/>
#     <node id="Media belonging to Scenario"/>
#     <node id="Decision Point Topic"/>
#     <node id="Topic 1 Decision Point Topic"/>
#     <node id="Topic 1 Step 1"/>
#     <node id="Media belonging to Topic 1 Step 1"/>
#     <node id="Topic 1 Step 2"/>
#     <node id="Media belonging to Topic 1 Step 2"/>
#     <node id="Quiz Topic 1 Decision Point Topic"/>
#     <node id="Question 1 Quiz Topic 1 Decision Point Topic"/>
#     <node id="Question 2 Quiz Topic 1 Decision Point Topic"/>
#     <node id="Topic 2 Decision Point Topic"/>
#     <node id="Topic 2 Step 1"/>
#     <node id="Media belonging to Topic 2 Step 1"/>
#     <node id="Topic 2 Step 2"/>
#     <node id="Media belonging to Topic 2 Step 2"/>
#     <node id="Question 1 Topic 2 Step 2"/>
#     <node id="Topic 2 Step 3"/>
#     <node id="Media belonging to Topic 2 Step 3"/>
#     <node id="Topic 2 Step 4"/>
#     <node id="Media belonging to Topic 2 Step 4"/>
#     <node id="Quiz Topic 2 Decision Point Topic"/>
#     <node id="Question 1 Quiz Topic 2 Decision Point Topic"/>
#     <node id="Question 2 Quiz Topic 2 Decision Point Topic"/>
#     <node id="Question 3 Quiz Topic 2 Decision Point Topic"/>
#     <!-- Edges -->
#     <edge source="Topic" target="Topic">
#     <data key="d0">Introduction to Renewable Energy</data>
#     </edge>
#     <edge source="Topic" target="Scenario">
#     <data key="d0">The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.</data>
#     </edge>
#     <edge source="Scenario" target="Media belonging to Scenario">
#     <data key="d0"> Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.</data>
#     </edge>
#     <edge source="Scenario" target="Decision Point Topic">
#     <data key="d0">Choose a renewable energy source to explore how it works and its benefits.</data>
#     </edge>
#     <edge source="Decision Point Topic" target="Topic 1 Decision Point Topic">
#     <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
#     </edge>
#     <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 1">
#     <data key="d0">Understanding how wind turbines convert wind into electricity</data>
#     </edge>
#     <edge source="Topic 1 Step 1" target="Media belonging to Topic 1 Step 1">
#     <data key="d0"> Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.</data>
#     </edge>
#     <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 2">
#     <data key="d0">The environmental impact and benefits of wind energy</data>
#     </edge>
#     <edge source="Topic 1 Step 2" target="Media belonging to Topic 1 Step 2">
#     <data key="d0"> Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.</data>
#     </edge>
#     <edge source="Topic 1 Step 2" target="Quiz Topic 1 Decision Point Topic">
#     <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 1 Decision Point Topic)?</data>
#     </edge>
#     <edge source="Quiz Topic 1 Decision Point Topic" target="Question 1 Quiz Topic 1 Decision Point Topic">
#     <data key="d0">What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points</data>
#     </edge>
#     <edge source="Quiz Topic 1 Decision Point Topic" target="Question 2 Quiz Topic 1 Decision Point Topic">
#     <data key="d0">True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points</data>
#     </edge>
#     <edge source="Decision Point Topic" target="Topic 2 Decision Point Topic">
#     <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
#     </edge>
#     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 1">
#     <data key="d0">How solar panels convert sunlight into electrical energy</data>
#     </edge>
#     <edge source="Topic 2 Step 1" target="Media belonging to Topic 2 Step 1">
#     <data key="d0"> Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.</data>
#     </edge>
#     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 2">
#     <data key="d0">The role of solar energy in powering homes and businesses</data>
#     </edge>
#     <edge source="Topic 2 Step 2" target="Media belonging to Topic 2 Step 2">
#     <data key="d0"> Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.</data>
#     </edge>
#     <edge source="Topic 2 Step 2" target="Question 1 Topic 2 Step 2">
#     <data key="d0">What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic</data>
#     </edge>
#     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 3">
#     <data key="d0">Installation and maintenance of solar panel systems</data>
#     </edge>
#     <edge source="Topic 2 Step 3" target="Media belonging to Topic 2 Step 3">
#     <data key="d0"> Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.</data>
#     </edge>
#     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 4">
#     <data key="d0">Future innovations in solar technology</data>
#     </edge>
#     <edge source="Topic 2 Step 4" target="Media belonging to Topic 2 Step 4">
#     <data key="d0"> Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.</data>
#     </edge>
#     <edge source="Topic 2 Step 4" target="Quiz Topic 2 Decision Point Topic">
#     <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 2 Decision Point Topic)?</data>
#     </edge>
#     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 1 Quiz Topic 2 Decision Point Topic">
#     <data key="d0">Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points</data>
#     </edge>
#     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 2 Quiz Topic 2 Decision Point Topic">
#     <data key="d0">True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points</data>
#     </edge>
#     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 3 Quiz Topic 2 Decision Point Topic">
#     <data key="d0">Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points</data>
#     </edge>
# </graph>
# </graphml>'''

                # return Response(chain(usr_inp), mimetype='text/plain')
                return Response(chain(usr_inp), mimetype='text/plain')
            
                # return redirect(url_for('stream_template',usr_inp=text_data)) 

                # output_graphml = LCD.GENERATE_GRAPHML(text_data)
                # with open(path_graphml, 'w', encoding='utf-8') as graphml_file:
                #     graphml_file.write(output_graphml)

                # graphml_content = output_graphml
                # return jsonify({'graphml_content': output_graphml}) #This line sends to index_copy the 'graphml_content' 
            elif width and height and graphcode:
                print("elif width and height and graphcode")
                with open(path_graphml, 'w', encoding='utf-8') as graphml_file:
                    graphml_file.write(graphcode)
                print(width,height,graphcode,"GOT IT!")
                with open(path_graphml, 'r') as graphml_file:
                    graphml_content = graphml_file.read()
                    if width and height and graphml_content:
                        try:
                            plot_image_uri = LCD.DRAW_GRAPH(path_graphml, width, height)
                            # flash("The GraphML generated was compilable!")
                            return jsonify({'img_uri': plot_image_uri,'msg':'The GraphML generated was compilable!'})
                        except:
                            return jsonify({'msg':'The GraphML is not compilable, Go back to Regenerate!'})
                            # flash("The GraphML is not compilable, Go back to Regenerate!")
            else:
                print("No text data provided")
                return jsonify({'error': 'No text data provided'}), 400
        else:  # If request contains form data
            text_datas = request.form.get('textData')
    # else:
    #     print("GETING!!!!!")
    #     data = request.json
    #     graphcode = data.get('graphcode')
    #     with open(path_graphml, 'w', encoding='utf-8') as graphml_file:
    #         graphml_file.write(graphcode)

    #     with open(path_graphml, 'r') as graphml_file:
    #         graphml_content = graphml_file.read()

    #     width = request.args.get('width')
    #     height = request.args.get('height')
    #     # graphcode = data.get('graphcode')
    #     print(width,height,graphml_content)
    #     if width and height and graphml_content:
    #         try:
    #             #plot_image_uri = LCD.DRAW_GRAPH(path_graphml, width, height)
    #             plot_image_uri = LCD.DRAW_GRAPH(path_graphml, width, height)
    #             flash("The GraphML generated was compilable!")

    #         except:
    #             flash("The GraphML is not compilable, Go back to Regenerate!")
    
    return render_template("index_copy.html", debug_output_graphml=graphml_content, img_uri=plot_image_uri, text_data_graphtext=text_datas)

# Route to stream the template (DANTE method)
# @app.route('/stream_template')
# def stream_template():
#     usr_inp = request.args.get('usr_inp', default='')
#     class ThreadedGenerator:
#         def __init__(self):
#             self.queue = queue.Queue()

#         def __iter__(self):
#             return self

#         def __next__(self):
#             item = self.queue.get()
#             if item is StopIteration: raise item
#             return item

#         def send(self, data):
#             self.queue.put(data)

#         def close(self):
#             self.queue.put(StopIteration)

#     class ChainStreamHandler(StreamingStdOutCallbackHandler):
#         def __init__(self, gen):
#             super().__init__()
#             self.gen = gen

#         def on_llm_new_token(self, token: str, **kwargs):
#             print(token)
#             self.gen.send(token)

#     def llm_thread(g,prompt):
#         try:
#             llmsx = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True, callbacks=[ChainStreamHandler(g)],verbose=True)
#             graphml_chain = LCD.GENERATE_GRAPHML(prompt,llmsx)
#             graphml_chain.run(text=prompt)
#         finally:
#             g.close()

#     def chain(prompt):
#         if prompt != "":
#             g = ThreadedGenerator()
#             threading.Thread(target=llm_thread, args=(g,prompt)).start()
#             return g
# #     graphmlml='''<?xml version="1.0" encoding="UTF-8"?>
# # <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
# #     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
# #     xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
# #     http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
# # <key id="d0" for="edge" attr.name="description" attr.type="string"/>
# # <graph id="G" edgedefault="undirected">
# #     <!-- Nodes -->
# #     <node id="Topic"/>
# #     <node id="Scenario"/>
# #     <node id="Media belonging to Scenario"/>
# #     <node id="Decision Point Topic"/>
# #     <node id="Topic 1 Decision Point Topic"/>
# #     <node id="Topic 1 Step 1"/>
# #     <node id="Media belonging to Topic 1 Step 1"/>
# #     <node id="Topic 1 Step 2"/>
# #     <node id="Media belonging to Topic 1 Step 2"/>
# #     <node id="Quiz Topic 1 Decision Point Topic"/>
# #     <node id="Question 1 Quiz Topic 1 Decision Point Topic"/>
# #     <node id="Question 2 Quiz Topic 1 Decision Point Topic"/>
# #     <node id="Topic 2 Decision Point Topic"/>
# #     <node id="Topic 2 Step 1"/>
# #     <node id="Media belonging to Topic 2 Step 1"/>
# #     <node id="Topic 2 Step 2"/>
# #     <node id="Media belonging to Topic 2 Step 2"/>
# #     <node id="Question 1 Topic 2 Step 2"/>
# #     <node id="Topic 2 Step 3"/>
# #     <node id="Media belonging to Topic 2 Step 3"/>
# #     <node id="Topic 2 Step 4"/>
# #     <node id="Media belonging to Topic 2 Step 4"/>
# #     <node id="Quiz Topic 2 Decision Point Topic"/>
# #     <node id="Question 1 Quiz Topic 2 Decision Point Topic"/>
# #     <node id="Question 2 Quiz Topic 2 Decision Point Topic"/>
# #     <node id="Question 3 Quiz Topic 2 Decision Point Topic"/>
# #     <!-- Edges -->
# #     <edge source="Topic" target="Topic">
# #     <data key="d0">Introduction to Renewable Energy</data>
# #     </edge>
# #     <edge source="Topic" target="Scenario">
# #     <data key="d0">The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.</data>
# #     </edge>
# #     <edge source="Scenario" target="Media belonging to Scenario">
# #     <data key="d0"> Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.</data>
# #     </edge>
# #     <edge source="Scenario" target="Decision Point Topic">
# #     <data key="d0">Choose a renewable energy source to explore how it works and its benefits.</data>
# #     </edge>
# #     <edge source="Decision Point Topic" target="Topic 1 Decision Point Topic">
# #     <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
# #     </edge>
# #     <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 1">
# #     <data key="d0">Understanding how wind turbines convert wind into electricity</data>
# #     </edge>
# #     <edge source="Topic 1 Step 1" target="Media belonging to Topic 1 Step 1">
# #     <data key="d0"> Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.</data>
# #     </edge>
# #     <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 2">
# #     <data key="d0">The environmental impact and benefits of wind energy</data>
# #     </edge>
# #     <edge source="Topic 1 Step 2" target="Media belonging to Topic 1 Step 2">
# #     <data key="d0"> Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.</data>
# #     </edge>
# #     <edge source="Topic 1 Step 2" target="Quiz Topic 1 Decision Point Topic">
# #     <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 1 Decision Point Topic)?</data>
# #     </edge>
# #     <edge source="Quiz Topic 1 Decision Point Topic" target="Question 1 Quiz Topic 1 Decision Point Topic">
# #     <data key="d0">What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points</data>
# #     </edge>
# #     <edge source="Quiz Topic 1 Decision Point Topic" target="Question 2 Quiz Topic 1 Decision Point Topic">
# #     <data key="d0">True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points</data>
# #     </edge>
# #     <edge source="Decision Point Topic" target="Topic 2 Decision Point Topic">
# #     <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
# #     </edge>
# #     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 1">
# #     <data key="d0">How solar panels convert sunlight into electrical energy</data>
# #     </edge>
# #     <edge source="Topic 2 Step 1" target="Media belonging to Topic 2 Step 1">
# #     <data key="d0"> Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.</data>
# #     </edge>
# #     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 2">
# #     <data key="d0">The role of solar energy in powering homes and businesses</data>
# #     </edge>
# #     <edge source="Topic 2 Step 2" target="Media belonging to Topic 2 Step 2">
# #     <data key="d0"> Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.</data>
# #     </edge>
# #     <edge source="Topic 2 Step 2" target="Question 1 Topic 2 Step 2">
# #     <data key="d0">What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic</data>
# #     </edge>
# #     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 3">
# #     <data key="d0">Installation and maintenance of solar panel systems</data>
# #     </edge>
# #     <edge source="Topic 2 Step 3" target="Media belonging to Topic 2 Step 3">
# #     <data key="d0"> Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.</data>
# #     </edge>
# #     <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 4">
# #     <data key="d0">Future innovations in solar technology</data>
# #     </edge>
# #     <edge source="Topic 2 Step 4" target="Media belonging to Topic 2 Step 4">
# #     <data key="d0"> Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.</data>
# #     </edge>
# #     <edge source="Topic 2 Step 4" target="Quiz Topic 2 Decision Point Topic">
# #     <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 2 Decision Point Topic)?</data>
# #     </edge>
# #     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 1 Quiz Topic 2 Decision Point Topic">
# #     <data key="d0">Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points</data>
# #     </edge>
# #     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 2 Quiz Topic 2 Decision Point Topic">
# #     <data key="d0">True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points</data>
# #     </edge>
# #     <edge source="Quiz Topic 2 Decision Point Topic" target="Question 3 Quiz Topic 2 Decision Point Topic">
# #     <data key="d0">Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points</data>
# #     </edge>
# # </graph>
# # </graphml>'''

#     # return Response(chain(usr_inp), mimetype='text/plain')
#     return Response(chain(usr_inp), mimetype='text/plain')

if __name__ == '__main__':
    app.run()

