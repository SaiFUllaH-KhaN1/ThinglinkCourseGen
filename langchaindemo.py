import soundfile as sf
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import BaseChatPromptTemplate, PromptTemplate
from langchain.agents import initialize_agent, Tool, load_tools, AgentType
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from reportlab.pdfgen import canvas 
import os
from flask import Flask, render_template, request, session, flash, get_flashed_messages
from io import BytesIO
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
load_dotenv(dotenv_path="E:\downloads\THINGLINK\Chat Foundation HTML 05 Jan DEMO - Copy\HUGGINGFACEHUB_API_TOKEN.env")

# llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
# template = """You are a chatbot having a conversation with a human.

# {chat_history}
# Human: {user_input}
# Chatbot:"""

# def llm_conversation(user_input):
#     response_bot = chain.predict(user_input=user_input)
#     return response_bot

def RAG(pdf_file):
    # read in your pdf file
    pdf_reader = PdfReader(pdf_file)

    # read data from the file and put them into a variable called text
    text = ''
    for i, page in enumerate(pdf_reader.pages):
        text_instant = page.extract_text()
        if text_instant:
            text += text_instant
    # chunking
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap  = 64,
    length_function = len,
    )
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    print("docsearch made")
    return docsearch


prompt = PromptTemplate(
    input_variables=["human_input","subject_name","chat_history"],
    template=""" Show the answer to human's input in step-by-step instruction format such that you are instructing and
    teaching a student. The instructions should be clear, descriptive and very accurate. Before each instruction mention step number.

    Make sure to include the images tags after each step-by-step instructions.
    The image tags should specify to what step number they belong and also mention the subject name given in {subject_name}
    to describe what the image is for and also give instructions on how and where to shoot these images
    so that the course instructor can also give the student illustrated course with images where they
    can learn how to physically apply their knowledge. Provide image captions in following format strictly enclosed in square brackets:
    [ Image belonging to Step Number, Subject_name from {subject_name}: Description of image and give mandatory instructions on shooting them to describe the instruction step ]

    At the end you must formulate a quiz of MCQ type with few questions that can enable the course instructor to test student's knowledge.
    The number of questions in the quiz are relevant to number of important critical steps such that without the information of these steps
    the task at hand is not achievable. 
    While making the quiz, make sure the questions are only related to critical important steps involved in
    the step-by-step instructions you have provided earlier. You absolutley have to 
    give score of each quiz question. Total score of all the quiz questions must accumulate to 100
    and it is up to you to give score of each quiz question according to it's importance.
    Remember to give correct answers of the questions at the very end.

    You also have the context and history of conversation at here: {chat_history}. This will help you to
    remember past conversations with the human and you can edit your provided responses accordingly
    to make the course better and more specific to the human's requirement.
    \n\nEXAMPLE FOR YOUR UNDERSTANDING\n\n
    Human: Removing the power-button board
    Output:\n
    To remove the power-button board, follow these steps:

    Step 1: Open the hinge to an angle of 90 degrees. 
    [Image belonging to Step 1, Inspiron 15 3000: Take a picture of the laptop with the hinge opened at a 90-degree angle]

    Step 2: Open the latch and disconnect the power-button board cable from the system board. 
    [Image belonging to Step 2, Inspiron 15 3000: Take a close-up picture of the latch and the cable connection]

    Step 3: Remove the screw (M2x2) that secures the power-button board to the palm-rest assembly. 
    [Image belonging to Step 3, Inspiron 15 3000: Take a picture of the screw that needs to be removed]

    Step 4: Peel the tape that secures the power-button board to the palm-rest assembly. 
    [Image belonging to Step 4, Inspiron 15 3000: Take a picture of the tape being peeled off]

    Step 5: Slide and remove the power-button board from the tab on the palm-rest assembly. 
    [Image belonging to Step 5, Inspiron 15 3000: Take a picture of the power-button board being slid and removed]

    Step 6: Note the power-button board cable routing and peel it off the palm-rest assembly. 
    [Image belonging to Step 6, Inspiron 15 3000: Take a picture of the cable routing and the process of peeling it off]

    Please make sure to follow the safety instructions provided in the context before working inside your computer.

    Now, let's move on to the quiz to test your knowledge:

    Quiz:
    1. What is the first step to remove the power-button board?
    a) Disconnect the power-button board cable
    b) Open the hinge to an angle of 90 degrees
    c) Remove the screw that secures the power-button board
    d) Peel the tape that secures the power-button board
    Correct Answer Quiz Question 1: b) Open the hinge to an angle of 90 degrees

    2. How should the power-button board be removed from the palm-rest assembly?
    a) Slide and remove it from the tab
    b) Lift it straight up
    c) Twist it counterclockwise
    d) Push it downwards
    Correct Answer Quiz Question 2: a) Slide and remove it from the tab

    3. What should you do with the power-button board cable after removing the board?
    a) Leave it connected to the system board
    b) Disconnect it from the system board
    c) Peel it off the palm-rest assembly
    d) Wrap it around the power-button board
    Correct Answer Quiz Question 3: c) Peel it off the palm-rest assembly

    4. What type of screw secures the power-button board to the palm-rest assembly?
    a) M2x2
    b) M2x3
    c) 2.5x8
    d) 3x6
    Correct Answer Quiz Question 4: a) M2x2

    5. What should you do before working inside your computer?
    a) Read the safety information
    b) Disconnect all cables
    c) Remove the battery
    d) Wear gloves
    Correct Answer Quiz Question 5: a) Read the safety information

    Quiz Question 1 Score: 20 points
    Quiz Question 2 Score: 20 points
    Quiz Question 3 Score: 20 points
    Quiz Question 4 Score: 20 points
    Quiz Question 5 Score: 20 points

    \n\nEND OF EXAMPLE\n\nThe above example is just for understanding of the format you should adhere with. It does not
    mean that you have to give response with 5 quiz questions or fix amount of steps as written in the example.
    You can give instruction, image tags and quiz questions in numbers that are suitable to the information.
    Human: {human_input}
    Chatbot:"""
)



# chain = load_qa_chain(
#     llm=llm, chain_type="stuff", prompt=prompt
# )


def TALK_WITH_RAG(query, docsearch, llm):
    print("TALK_WITH_RAG Initiated!")
    docs = docsearch.similarity_search(query, k=3)
    # chain = load_qa_chain(
    #     llm=llm, chain_type="stuff", prompt=prompt
    # )
    chain = LLMChain(prompt=prompt, llm=llm)
    ### Static Query###   
    docs_page_contents = [doc.page_content for doc in docs]
    docs_whole_contents = docsearch.similarity_search("Title name of object, device or theory of this document", k=1)
    static_query = """For what object, device or theory is this document written for? Only write the short title name for it.
    Use information obtained from user relevant specific search results {docs_page_contents} and the general document {docs_whole_contents}, to 
    give a short title name suggestion and do not describe anything."""

    llm_title = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=32)
    prompt_title = PromptTemplate(
        input_variables=["docs_whole_contents","docs_page_contents"],
        template=static_query)
    llm_title_chain = LLMChain(prompt=prompt_title, llm=llm_title)
    title_name_output = llm_title_chain.run({"docs_whole_contents": docs_whole_contents, "docs_page_contents": docs_page_contents})
    print(title_name_output)
    # subject_name = title_name_output
    from langchain.chains import create_extraction_chain
    # Schema
    schema_title = {
        "properties": {
            "subject_name": {"type": "string"},
        },
        "required": ["subject_name"],
    }
    # Run chain
    llm_title_extraction_chain = create_extraction_chain(schema_title, llm_title)
    subject_name = llm_title_extraction_chain.run(title_name_output)
    print(subject_name)   
    print(docs)
    return chain, docs, query, subject_name


def GENERATE_GRAPHML(bot_last_reply):
    print("This is last reply",bot_last_reply)
    GRAPHML_PROMPT = PromptTemplate(input_variables=['text'], template="""You are a networked intelligence helping a human track knowledge by giving providing them with
    graphml having various nodes and edges representing information about all
    relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    Extract all of the nodes and edges from the text. The nodes are connected to relevant other nodes by edges. These edges have also information in them.
    The concept is like a knowledge triple, which is a clause that contains a subject, a predicate, and an object.
    The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.
    You are only allowed to form the node ids of "Topic", "Step (number)", "Step (number) instructions",
    "Image tag belonging to Step (number)" and "Quiz question (number)".
    \n\nEXAMPLE\n\nTo remove the solid-state drive, please follow these step-by-step instructions:
    Step 1: Remove the screw
    - Locate the screw (M2x3) that secures the solid-state drive to the solid-state screw mount.
    - Use a screwdriver to remove the screw.
    [Image belonging to Step 1, Inspiron 15 3000: Take a close-up image of the screw that needs to be removed.]

    Step 2: Slide and remove the solid-state drive
    - Slide the solid-state drive out of the solid-state drive slot.
    - Carefully remove the solid-state drive from the slot.
    [Image belonging to Step 2, Inspiron 15 3000: Take an image showing the solid-state drive being slid out of the slot.]

    Step 3: Lift off the solid-state drive bracket
    - Lift the solid-state drive off the solid-state drive bracket.
    - Make sure to handle the solid-state drive with care as it is fragile.
    [Image belonging to Step 3, Inspiron 15 3000: Take an image showing the solid-state drive being lifted off the bracket.]

    Congratulations! You have successfully removed the solid-state drive from your Inspiron 15 3000 laptop.

    Now, let's move on to the quiz to test your knowledge:

    Quiz:
    1. What type of screw secures the solid-state drive to the solid-state screw mount?
    a) M2x3
    b) M3x4
    c) M4x5
    d) M5x6
    Correct Answer Quiz Question 1: b) M3x4
                                    
    2. How should you handle the solid-state drive during the removal process?
    a) With gloves
    b) With bare hands
    c) With a screwdriver
    d) With care
    Correct Answer Quiz Question 1: d) With care
                                    
    Quiz Question 1 Score: 50 points
    Quiz Question 2 Score: 50 points

    Total Score: 100 points                                                                
    Output:\n
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d0" for="edge" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="undirected">
        <node id="Topic"/>
        <node id="Step 1"/>
        <node id="Step 1 instructions"/>
        <node id="Image tag belonging to Step 1"/>
        <node id="Quiz question 1"/>
        <node id="Step 2"/>
        <node id="Step 2 instructions"/>
        <node id="Image tag belonging to Step 2"/>
        <node id="Step 3"/>
        <node id="Step 3 instructions"/>
        <node id="Image tag belonging to Step 3"/>
        <node id="Quiz question 2"/>
        <edge source="Topic" target="Step 1">
        <data key="d0">To remove the solid-state drive</data>
        </edge>
        <edge source="Topic" target="Step 2">
        <data key="d0">To remove the solid-state drive</data>
        </edge>
        <edge source="Topic" target="Step 3">
        <data key="d0">To remove the solid-state drive</data>
        </edge>
        <edge source="Step 1" target="Step 1 instructions">
        <data key="d0">Remove the screw - Locate the screw (M2x3) that secures the solid-state drive to the solid-state screw mount. - Use a screwdriver to remove the screw.</data>
        </edge>
        <edge source="Step 2" target="Step 2 instructions">
            <data key="d0">Slide and remove the solid-state drive - Slide the solid-state drive out of the solid-state drive slot. - Carefully remove the solid-state drive from the slot.</data>
        </edge>
        <edge source="Step 3" target="Step 3 instructions">
            <data key="d0">Lift off the solid-state drive bracket - Lift the solid-state drive off the solid-state drive bracket. - Make sure to handle the solid-state drive with care as it is fragile.</data>
        </edge>                                                                        
        <edge source="Step 1" target="Image tag belonging to Step 1">
            <data key="d0">[Image belonging to Step 1, Inspiron 15 3000: Take a close-up image of the screw that needs to be removed.]</data>
        </edge>
        <edge source="Step 2" target="Image tag belonging to Step 2">
        <data key="d0">[Image belonging to Step 2, Inspiron 15 3000: Take an image showing the solid-state drive being slid out of the slot.]</data>
        </edge>
        <edge source="Step 3" target="Image tag belonging to Step 3">
            <data key="d0">[Image belonging to Step 3, Inspiron 15 3000: Take an image showing the solid-state drive being lifted off the bracket.]</data>
        </edge>                                                                        
        <edge source="Step 1" target="Quiz question 1">
        <data key="d0">1. What type of screw secures the solid-state drive to the solid-state screw mount?
    a) M2x3
    b) M3x4
    c) M4x5
    d) M5x6
    Correct Answer Quiz Question 1: b) M2x3</data>
        </edge>
        <edge source="Step 3" target="Quiz question 2">
            <data key="d0">2. How should you handle the solid-state drive during the removal process?
    a) With gloves
    b) With bare hands
    c) With a screwdriver
    d) With care
    Correct Answer Quiz Question 2: d) With care</data>
        </edge>
        <edge source="Quiz question 1" target="Quiz Question 1 Score">
        <data key="d0">50 points</data>
        </edge>
        <edge source="Quiz question 2" target="Quiz Question 2 Score">
        <data key="d0">50 points</data>
        </edge>                                                                                                                                                                       
    </graph>
    </graphml>
    \n\nEND OF EXAMPLE\n\n Please note that you absolutely should not form connections of source and target between steps( <edge source="Step 5" target="Step 6"> ), because
    we only want the topic to be connected to steps and the steps are further connected to their own relative available step instructions, quiz questions and image tags.
    Moreover, it is absolutley mandatory and necessary for you to generate a complete graphml response such that the Graphml generated from you must close by "</graph> </graphml>" at the end of your response
    and all it's edges and nodes are also closed in the required syntax rules of graphml and all the step instructions, image tags and quiz questions be included in it since we want our graphml
    to be compilable.   
    \n\n{text}Output:""")
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    graphml_chain = LLMChain(llm= llm, prompt=GRAPHML_PROMPT)
    output_graphml = graphml_chain.predict(text=bot_last_reply)

    return output_graphml

def DRAW_GRAPH(output_graphml_generated_again, width,height):
    import networkx as nx
    import matplotlib.pyplot as plt
    import base64
    import io
    G = nx.read_graphml(output_graphml_generated_again)
    plt.figure(figsize=(int(width),int(height)))
    options = {
        "font_size": 6,
        "linewidths": 3,
    }
    nx.draw_networkx(G, **options)
    ax = plt.gca()
    ax.margins(0.2)
    plt.plot()

    # Saving the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Converting the plot image to base64 for embedding in HTML
    plot_image = base64.b64encode(buffer.getvalue())
    plot_image_uri = f"data:image/png;base64,{plot_image.decode()}"

    return plot_image_uri

