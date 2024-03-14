import soundfile as sf
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import BaseChatPromptTemplate, PromptTemplate
from langchain.agents import initialize_agent, Tool, load_tools, AgentType
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from reportlab.pdfgen import canvas 
import matplotlib
import os
from flask import Flask, render_template, request, session, flash, get_flashed_messages
from io import BytesIO
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder
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


# prompt = PromptTemplate(
#     input_variables=["input_documents","human_input","subject_name","chat_history"],
#     template=""" Show the answer to human's input in step-by-step instruction format such that you are instructing and
#     teaching a student. The instructions should be clear, descriptive and very accurate 
#     based on the information present in {input_documents}; which is relevant to the human's 
#     input prompt or query: {human_input}. Look for Before each instruction mention step number.

#     Make sure to include the images tags after each step-by-step instructions.
#     The image tags should specify to what step number they belong and also mention the subject name given in {subject_name}
#     to describe what the image is for and also give instructions on how and where to shoot these images
#     so that the course instructor can also give the student illustrated course with images where they
#     can learn how to physically apply their knowledge. Provide image captions in following format strictly enclosed in square brackets:
#     [ Image belonging to Step Number, Subject_name from {subject_name}: Description of image and give mandatory instructions on shooting them to describe the instruction step ]

#     At the end you must formulate a quiz of MCQ type with few questions that can enable the course instructor to test student's knowledge.
#     The number of questions in the quiz are relevant to number of important critical steps such that without the information of these steps
#     the task at hand is not achievable. 
#     While making the quiz, make sure the questions are only related to critical important steps involved in
#     the step-by-step instructions you have provided earlier. You absolutley have to 
#     give score of each quiz question. Total score of all the quiz questions must accumulate to 100
#     and it is up to you to give score of each quiz question according to it's importance.
#     Remember to give correct answers of the questions at the very end.

#     You also have the context and history of conversation at here: {chat_history}. This will help you to
#     remember past conversations with the human and you can edit your provided responses accordingly
#     to make the course better and more specific to the human's requirement.
#     \n\nEXAMPLE FOR YOUR UNDERSTANDING\n\n
#     Human: Removing the power-button board
#     Output:\n
#     To remove the power-button board, follow these steps:

#     Step 1: Open the hinge to an angle of 90 degrees. 
#     [Image belonging to Step 1, Inspiron 15 3000: Take a picture of the laptop with the hinge opened at a 90-degree angle]

#     Step 2: Open the latch and disconnect the power-button board cable from the system board. 
#     [Image belonging to Step 2, Inspiron 15 3000: Take a close-up picture of the latch and the cable connection]

#     Step 3: Remove the screw (M2x2) that secures the power-button board to the palm-rest assembly. 
#     [Image belonging to Step 3, Inspiron 15 3000: Take a picture of the screw that needs to be removed]

#     Step 4: Peel the tape that secures the power-button board to the palm-rest assembly. 
#     [Image belonging to Step 4, Inspiron 15 3000: Take a picture of the tape being peeled off]

#     Step 5: Slide and remove the power-button board from the tab on the palm-rest assembly. 
#     [Image belonging to Step 5, Inspiron 15 3000: Take a picture of the power-button board being slid and removed]

#     Step 6: Note the power-button board cable routing and peel it off the palm-rest assembly. 
#     [Image belonging to Step 6, Inspiron 15 3000: Take a picture of the cable routing and the process of peeling it off]

#     Please make sure to follow the safety instructions provided in the context before working inside your computer.

#     Now, let's move on to the quiz to test your knowledge:

#     Quiz:
#     1. What is the first step to remove the power-button board?
#     a) Disconnect the power-button board cable
#     b) Open the hinge to an angle of 90 degrees
#     c) Remove the screw that secures the power-button board
#     d) Peel the tape that secures the power-button board
#     Correct Answer Quiz Question 1: b) Open the hinge to an angle of 90 degrees

#     2. How should the power-button board be removed from the palm-rest assembly?
#     a) Slide and remove it from the tab
#     b) Lift it straight up
#     c) Twist it counterclockwise
#     d) Push it downwards
#     Correct Answer Quiz Question 2: a) Slide and remove it from the tab

#     3. What should you do with the power-button board cable after removing the board?
#     a) Leave it connected to the system board
#     b) Disconnect it from the system board
#     c) Peel it off the palm-rest assembly
#     d) Wrap it around the power-button board
#     Correct Answer Quiz Question 3: c) Peel it off the palm-rest assembly

#     4. What type of screw secures the power-button board to the palm-rest assembly?
#     a) M2x2
#     b) M2x3
#     c) 2.5x8
#     d) 3x6
#     Correct Answer Quiz Question 4: a) M2x2

#     5. What should you do before working inside your computer?
#     a) Read the safety information
#     b) Disconnect all cables
#     c) Remove the battery
#     d) Wear gloves
#     Correct Answer Quiz Question 5: a) Read the safety information

#     Quiz Question 1 Score: 20 points
#     Quiz Question 2 Score: 20 points
#     Quiz Question 3 Score: 20 points
#     Quiz Question 4 Score: 20 points
#     Quiz Question 5 Score: 20 points

#     \n\nEND OF EXAMPLE\n\nThe above example is just for understanding of the format you should adhere with. It does not
#     mean that you have to give response with 5 quiz questions or fix amount of steps as written in the example.
#     You can give instruction, image tags and quiz questions in numbers that are suitable to the information.
#     Human: {human_input}
#     Chatbot:"""
# )
prompt = PromptTemplate(
    input_variables=["input_documents","human_input","subject_name"],#,"chat_history"],
    template="""
    You are an educational chat bot that helps in building training courses for human. You utilize a system
    where there are four scenarios. You prepare the courses adhering to the
    format of one of the four scenarios given as examples below, suitable to the type of information which
    is relevant to the human's input query or prompt. 

    The four scenarios and their introduction are Escape Room (useful for gamifying a certain learning
    situation such that human is given a scene to utilize the knowledge regarding a 
    certain subject to use it to escape from the scene. Self exploration, clue finding, investigative and critical
    thinking skills are aim of this scenario), Linear (useful in giving quick, straight-forward and easy training
    of a certain topic in a procedural format while also checking what user learned via questions and quiz questions at
    the end of the training.), Self Exploratory (useful in 
    giving a sandbox type of format where user is free to explore various aspects of a certain topic such that
    for a given topic, there could be branches of subtopics, each having procedural guide or information with quiz questions
    and the ability for user to navigate to move on to various subtopic after finishing a subtopic or move on to a quiz), Simulation (a gamifying approach to
    learning such that user is given choices to take for a certain topic or learning scenario, the choices leads to
    consequences. These consequences either takes you to other decision points where user is given more further choices
    or it may end the scenario with a result. There can be multiple results or outcomes for a given scenario).

    Here are the examples of each scenario. You have to absolutely make sure that it is
    mandatory for you to use either one of the scenarios below to formulate a course that adheres to the format of the selected
    scenario, especially the tags enclosed in single quotation marks '' by which the content is navigated, however, you are free to mold
    the amound of steps, decision points, quiz questions, choices etc., to the suitability and scale of the
    information about which human is inquiring you about. Stick to the format and tags of the specific 
    scenario you have selected and do not use tags and format of other scenarios once you selected a specfic scenario.  
    \n\nEXAMPLE FORMAT FOR EACH SCENARIO\n\n
    \nSIMULATION SCENARIO:\n
    'SIMULATION SCENARIO':
    'Learning Objectives':
    - Navigate through a corporate archive section to understand its layout and the function of different sub-sections.
    - Identify the operational processes within the printing office, including the use and benefits of specific equipment like scanners and printers.
    - Explore the management and correspondence functions within the head office, including team organization and documentation.
    - Investigate the library section to learn how publications are stored, accessed, and managed digitally and physically.
    - Apply decision-making skills to explore detailed aspects of archive management, such as digital storage solutions and publication cataloging.
    'Topic': Introduction to the Archive Section
    'Scenario': You are in the main lounge of a company building on your first day for an office tour. You are being shown an Archive Section where there different sub-sections situated at the left, right and upstairs from the main lounge.
    [Media belonging to 'Scenario', Description: A 360-degree image of a corporate lounge with paths leading to the left, right, and upstairs. Overlay Tags: Tag 1 - 'Left Path': An arrow pointing left with a caption: "To the Printing Office". Tag 2 - 'Right Path': An arrow pointing right with a caption: "To the Library". Tag 3 - 'Upstairs': An arrow pointing upwards with a caption: "To the Head Office".]
    'Decision Point Main': There are 3 paths to choose from the main lounge. Left, Right and Upstairs.
    'Timer Decision Point Main': 1 minutes
    'Choice 1 Decision Point Main': Left path.
    'Choice 1 Consequence Decision Point Main': You reach the printing office of Archive Section when you move left from the lounge. 'Move on to Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point Main', Description: Picture of an office with various printers and a large dedicated scanner. Overlay Tags: Tag 1 - 'Printer': Close-up on a printer with a caption: "60 Copies per Minute". Tag 2 - 'Dedicated Scanner': Close-up on a scanner with a caption: "Scans up to A0 size paper".]
    'Choice 2 Decision Point Main': Upstairs.
    'Choice 2 Consequence Decision Point Main': You reach to the head office of Archive Section where management and correspondence is carried out. 'Move on to Decision Point 2'.
    [Media belonging to 'Choice 2 Consequence Decision Point Main', Description: A picture showing an office that may belong to a manager, with documents and a computer on a desk. Overlay Tags: Tag 1 - 'Manager's Desk': A close-up on the desk with a caption: "Team State Document".]
    'Choice 3 Decision Point Main': Right path.
    'Choice 3 Consequence Decision Point Main': You reach the library of the Archive Section. 'Move on to Decision Point 3'.
    [Media belonging to 'Choice 3 Consequence Decision Point Main', Description: Picture of a library with shelves full of books and a computer setup for accessing records. Overlay Tags: Tag 1 - 'Computer Records': Zoom in on the computer screen with a caption: "Access Publication Records Here". Tag 2 - 'Soft Storage Cabinets': Focus on closed cabinets with a caption: "Contains Digital Archives".]

    'Decision Point 1': The printing office is used to print the publications. They are given to the library subsection after printing. There are 2 machines in the printing office, a large dedicated scanner and a large printer.
    'Timer Decision Point 1': 5 minutes
    'Choice 1 Decision Point 1': Dedicated Scanner.
    'Choice 1 Consequence Decision Point 1': The dedicated scanner can scan upto A0 size paper, mostly used for scanning large engineering blue-prints. 'Move on to Result Choice 1 Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point 1', Description: Video of a large drawing being scanned by a scanner. Overlay Tags: Tag 1 - 'Scanning Process': Detailed video showing how the dedicated scanner operates, including paper feeding, scanning mechanism, and output. Tag 2 - 'Blueprint Scanning': Tips on preparing and positioning large engineering blueprints for scanning.]
    'Result Choice 1 Decision Point 1': Score 5
    'Choice 2 Decision Point 1': Printer. 
    'Choice 2 Consequence Decision Point 1': This printer can print 60 Copies per Minute. 'Move on to Result Choice 2 Decision Point 1'.
    [Media belonging to 'Choice 2 Consequence Decision Point 1', Description: Video of a large printer in operation, showing rapid printing. Overlay Tags: Tag 1 - 'Printer Capabilities': Explanation of printer features, including speed, paper handling, and quality settings. Tag 2 - 'High-Volume Printing': Best practices for managing large print jobs efficiently.]
    'Result Choice 2 Decision Point 1': Score 5

    'Decision Point 2': The head office is used for correspondence, and management.
    'Timer Decision Point 2': 5 minutes
    'Choice 1 Decision Point 2': See the team state document.
    'Choice 1 Consequence Decision Point 2': The team state document shows all the employees and their data with work responsibilities and leave records. 'Move on to Result Choice 1 Decision Point 2'.
    [Media belonging to 'Choice 1 Consequence Decision Point 2', Description: Picture of an office desk with various documents and a computer screen displaying an employee database. Overlay Tags: Tag 1 - 'Employee Database': Detailed view of the database interface with functionalities highlighted. Tag 2 - 'Document Overview': Close-up on key documents that manage team responsibilities and leave schedules.]
    'Result Choice 1 Decision Point 2': Score 10

    'Decision Point 3': The library is used to keep all the record of publications printed in the printing section.
    'Timer Decision Point 3': 10 minutes
    'Choice 1 Decision Point 3': View computer records of publications.
    'Choice 1 Consequence Decision Point 3': The computer records contains record of available publications and their locations on the shelves. 'Move on to Result Choice 1 Decision Point 3'.
    [Media belonging to 'Choice 1 Consequence Decision Point 3', Description: An image of a computer screen displaying a digital catalog of publications with search functionality and shelf location information. Overlay Tags: Tag 1 - 'Digital Catalog': Interactive demo on navigating the digital records to find publications. Tag 2 - 'Locating Publications': Instructions on how to use the catalog to find the exact shelf location of a book or document.]
    'Result Choice 1 Decision Point 3': Score 5
    'Choice 2 Decision Point 3': View the closed soft storage cabinets.
    'Choice 2 Consequence Decision Point 3': The cabinet is opened for you to see inside. 'Move on to Decision Point 4'.
    [Media belonging to 'Choice 2 Consequence Decision Point 3', Description: Picture of an opened storage cabinet revealing various labeled compartments and digital storage media. Overlay Tags: Tag 1 - 'Storage Organization': Overview of how the cabinet is organized for efficient storage and retrieval. Tag 2 - 'Digital Media Storage': Insights into the preservation and categorization of digital archives.]

    'Decision Point 4': The cabinets contains various disks that stores the content of the publications in soft form, to be given with the publications at the time of issue to customers.
    'Timer Decision Point 4': 3 minutes
    'Choice 1 Decision Point 4': Inspect a disk in the cabinet.
    'Choice 1 Consequence Decision Point 4': On the disk, information is given about what publication this disk belongs to and where the location of that publication in the shelves is. 'Move on to Result Choice 1 Decision Point 4'.
    [Media belonging to 'Choice 1 Consequence Decision Point 4', Description: A close-up of a disk with labels indicating the publication it accompanies and a map showing its shelf location in the library. Overlay Tags: Tag 1 - 'Disk Content Overview': A quick guide on the information provided on the disk, including the publication it belongs to. Tag 2 - 'Finding Publications': Instructions on using the disk's label to locate the physical publication in the library.]
    'Result Choice 1 Decision Point 4': Score 20

    \nSelf-Exploratory Scenario:\n
    'Self-Exploratory Scenario':
    'Learning Objectives':
    - Differentiate between various types of renewable energy sources, such as wind and solar energy.
    - Understand the basic mechanisms behind how wind turbines and solar panels generate electricity.
    - Recognize the environmental and societal benefits of transitioning to renewable energy sources.
    - Engage with interactive media to explore the technical and environmental aspects of renewable energies.
    - Apply critical thinking to assess the impact of renewable energy on reducing carbon footprint and greenhouse gas emissions.
    'Topic': Introduction to Renewable Energy
    'Scenario':
    The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.
    [Media belonging to 'Scenario', Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.]
    'Decision Point Topic': Choose a renewable energy source to explore how it works and its benefits.
    'Topic 1 'Decision Point Topic': Wind Energy
    'Topic 1 Step 1': Understanding how wind turbines convert wind into electricity.
    [Media belonging to 'Topic 1 Step 1': Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.]
    'Topic 1 Step 2': The environmental impact and benefits of wind energy.
    [Media belonging to 'Topic 1 Step 2': Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 1 Decision Point Topic)'?
    'Quiz Topic 1 Decision Point Topic':
    'Question 1 Quiz Topic 1 Decision Point Topic': What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points
    'Question 2 Quiz Topic 1 Decision Point Topic': True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points

    'Topic 2 Decision Point Topic': Solar Energy
    'Topic 2 Step 1': How solar panels convert sunlight into electrical energy.
    [Media belonging to 'Topic 2 Step 1': Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.]
    'Topic 2 Step 2': The role of solar energy in powering homes and businesses.
    [Media belonging to 'Topic 2 Step 2': Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.]
    'Question 1 Topic 2 Step 2': What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic
    'Topic 2 Step 3': Installation and maintenance of solar panel systems.
    [Media belonging to 'Topic 2 Step 3': Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.]
    'Topic 2 Step 4': Future innovations in solar technology.
    [Media belonging to 'Topic 2 Step 4': Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 2 Decision Point Topic)'?
    'Quiz Topic 2 Decision Point Topic':
    'Question 1 Quiz Topic 2 Decision Point Topic': Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points
    'Question 2 Quiz Topic 2 Decision Point Topic': True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points
    'Question 3 Quiz Topic 2 Decision Point Topic': Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points                                   
    
    \nLINEAR SCENARIO:\n
    'LINEAR SCENARIO':
    'Learning Objectives':
    - Identify the basic steps in tying shoe laces.
    - Understand the importance of starting with even laces for effective tying.
    - Learn to create and manipulate loops to form a basic shoe lace knot.
    - Recognize different styles of laces and their tying techniques through media interaction.
    - Apply knowledge through quiz questions to reinforce learning of the shoe lace tying process.
    'Topic': How to Tie Shoe Laces
    [Media belonging to 'Topic', Image Description: An array of shoes with different styles of laces displayed on a white background. Overlay Tags: Tag 1 - 'Flat Laces', Tag 2 - 'Round Laces', Tag 3 - 'Colorful Laces'. Each tag, when clicked, leads to a short video showcasing the lacing technique for that style.]
    'Point 1': Ensure your shoe laces are even on both sides. Hold the ends of your laces, making sure they are of equal length.
    [Media belonging to 'Point 1', Video Description: A step-by-step guide on evening out your shoe laces before beginning the tying process. Overlay Tags: Tag 1 - 'Holding Technique', shows the correct way to hold laces. Tag 2 - 'Measuring Evenness', demonstrates how to ensure laces are even.]
    'Point 2': Cross the right lace over the left, pulling it underneath and through the loop formed, creating a simple knot.
    'Question 1 Point 2': What is the first action in tying shoe laces?
    Options: A. Make a bow. B. Cross the laces. C. Tie a double knot.
    Correct Answer: B. Cross the laces.
    'Point 3': Create a loop with each lace, holding them up to form what looks like two bunny ears.
    [Media belonging to 'Point 3', Image Description: Close-up of hands holding up two loops in the laces against a neutral background. Overlay Tags: Tag 1 - 'Loop Formation', detailed instructions on creating perfect loops. Tag 2 - 'Bunny Ear Technique', tips for maintaining loop shape.]
    'Point 4': Cross the two loops, pulling one loop through the space between them to form a secure knot.
    [Media belonging to 'Point 4', Video Description: Demonstrating the technique for crossing loops and securing the knot. Overlay Tags: Tag 1 - 'Crossing Loops', shows the crossing action. Tag 2 - 'Securing the Knot', provides additional tips for knot tightness.]
    'Quiz':
    'Question 1 Quiz Point 1': Which point involves checking if the laces are even?
    Options: A. Point 1. B. Point 2. C. Point 3.
    Correct Answer: A. Point 1.
    Score: 5 points
    'Question 2 Quiz Point 2': What shape do your hands make with the laces in Point 3?
    Options: A. Straight line. B. Bunny ears. C. Circle.
    Correct Answer: B. Bunny ears.
    Score: 5 points
    'Question 3 Quiz Point 4': How do you secure the knot?
    Options: A. By tying the loops together. B. By making another basic knot. C. No need to secure, it's already done.
    Correct Answer: A. By tying the loops together.
    Score: 5 points
    'Total Score: 15 points'

    \nESCAPE ROOM SCENARIO:\n
    'ESCAPE ROOM SCENARIO':
    'Learning Objectives':
    - Recognize the importance of quick and informed decision-making during a fire emergency.
    - Identify fire safety protocols, including the avoidance of elevators and the use of staircases for evacuation.
    - Implement strategies to minimize smoke inhalation, such as covering your nose and mouth and staying low to the ground.
    - Evaluate the safest exit routes, understanding the difference between familiar exits and potentially dangerous shortcuts.
    - Practice locating and moving to designated assembly points post-evacuation for accountability and further instructions from emergency services.
    'Topic': Exiting the Building in a Fire Emergency
    'Introduction': You're in a multi-story building when an alarm sounds, signaling a fire emergency. Smoke is starting to fill the corridors, and it's imperative to leave the building as quickly and safely as possible. Your actions and decisions will determine your fate.
    [Media belonging to 'Introduction': Description: A 360-degree image of a smoke-filled corridor with emergency lights flashing. Overlay Tags: Tag 1 - 'Emergency Lighting': An arrow or glow around the emergency lights to guide the way. Tag 2 - 'Smoke Density': Visual cues on the smoke's density, suggesting lower areas have less smoke.]

    'Decision Point 1': Identify the safest initial route to exit the building.
    'Clue 1 Decision Point 1': Fire safety protocols suggest avoiding elevators during a fire.
    'Clue 2 Decision Point 1': Look for illuminated exit signs that indicate the pathway to safety.
    'Timer Decision Point 1': 2 minutes
    'Correct Choice Decision Point 1': Head towards the nearest staircase.
    'Correct Choice Consequence Decision Point 1': You find the staircase and start descending safely. 'Move on to Decision Point 2'.
    [Media belonging to 'Correct Choice Consequence Decision Point 1': Description: Picture of an illuminated 'Exit' sign above a stairwell. Overlay Tags: Tag 1 - 'Staircase Access': An arrow or circle highlighting the exit sign and the entrance to the staircase. Tag 2 - 'Fire Safety Tip': A note on why stairs are safer than elevators during a fire.]
    'Incorrect Choice Decision Point 1': Attempt to use the elevator.
    'Incorrect Choice Consequence Decision Point 1': The elevator is non-operational during the emergency, wasting precious time. 'Jump back to Decision Point 1'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 1': Description: Video clip showing an "Out of Service" message on the elevator panel. Overlay Tags: Tag 1 - 'Out of Service Notice': Close-up on the elevator panel showing the message. Tag 2 - 'Reminder': A note reminding that elevators should not be used during fires.]

    'Decision Point 2': Encounter smoke in the staircase. Choose how to proceed.
    'Clue 1 Decision Point 2': Smoke rises, so lower air might be clearer.
    'Clue 2 Decision Point 2': Covering your nose and mouth can help filter harmful particles.
    'Timer Decision Point 2': 2 minutes
    'Correct Choice Decision Point 2': Use a cloth to cover your nose and mouth, and proceed down the stairs, staying as low as possible.
    'Correct Choice Consequence Decision Point 2': You manage to breathe more easily and continue your descent. 'Move on to Decision Point 3'.

    [Media belonging to 'Correct Choice Consequence Decision Point 2': Description: Picture of a person using a cloth over their mouth, crouched low on stairs. Overlay Tags: Tag 1 - 'Proper Cloth Use': Instructions for covering nose and mouth effectively. Tag 2 - 'Stay Low Strategy': Demonstration of the low posture to stay below the smoke layer.]
    'Incorrect Choice Decision Point 2': Ignore the smoke and run through it as quickly as possible.
    'Incorrect Choice Consequence Decision Point 2': You cough and struggle to see, slowing your progress significantly. 'Jump back to Decision Point 2'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 2': Description: Video clip of a person coughing and struggling to navigate through smoke. Overlay Tags: Tag 1 - 'Effects of Smoke Inhalation': Visual on the impact of smoke on breathing and vision. Tag 2 - 'Warning': Advice against rushing through heavy smoke.]

    'Decision Point 3': Choose between two exits: a nearby door leading outside or continue down to the main exit.
    'Clue 1 for Decision Point 3': Not all doors lead to safe exits; some may lead to areas that are more dangerous.
    'Clue 2 for Decision Point 3': Familiar exits are generally safer but might be more crowded.
    'Timer Decision Point 3': 3 minutes
    'Correct Choice Decision Point 3': Continue down to the main exit you're familiar with.
    'Correct Choice Consequence Decision Point 3': Despite the crowd, you exit safely into the open air. 'Move on to Decision Point 4'.
    [Media belonging to 'Correct Choice Consequence Decision Point 3': Description: 360-degree image of the crowded but orderly evacuation at the main building exit. Overlay Tags: Tag 1 - 'Crowded Exit Strategy': Tips on navigating through a crowded exit safely. Tag 2 - 'Familiar Path': Highlighting the familiar route, reinforcing the choice for safety.]
    'Incorrect Choice Decision Point 3': Try the nearby door for a quicker escape.
    'Incorrect Choice Consequence Decision Point 3': The door leads to a dead-end, forcing you to turn back and lose time. 'Jump back to Decision Point 3'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 3': Description: Picture of a dead-end area with no exit, smoke building up. Overlay Tags: Tag 1 - 'Dead-End Warning': A caution symbol or text indicating the dead-end. Tag 2 - 'Smoke Accumulation': Visual cues on the increasing smoke, emphasizing the urgency to find the correct exit.]

    'Decision Point 4': After exiting, you must choose where to go next.
    'Clue 1 for Decision Point 4': Emergency services recommend moving away from the building to allow access for firefighters and ambulances.
    'Clue 2 for Decision Point 4': Designated assembly points are set up to account for everyone.
    'Timer Decision Point 4': 2 minutes
    'Correct Choice Decision Point 4': Head to the designated assembly point.
    'Correct Choice Consequence Decision Point 4': You are accounted for and receive instructions from emergency personnel. 'Move on to Escape Block'.
    [Media belonging to 'Correct Choice Consequence Decision Point 4': Description: Picture of survivors gathering at the designated assembly point with emergency services in attendance. Overlay Tags: Tag 1 - 'Assembly Point Location': Arrows or markers pointing to the assembly area. Tag 2 - 'Emergency Instructions': A depiction of emergency personnel providing guidance to evacuees.]
    'Incorrect Choice Decision Point 4': Stay close to the building to watch what happens.
    'Incorrect Choice Consequence Decision Point 4': Emergency personnel usher you away for your safety, delaying their work. 'Jump back to Decision Point 4'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 4': Description: Video clip of emergency personnel directing people away from the building. Overlay Tags: Tag 1 - 'Safety Direction': Visual of emergency personnel guiding evacuees to safety. Tag 2 - 'Hazard Area': Highlighting the danger zone to avoid near the building.]

    'Escape Block': Congratulations, you've navigated through the fire emergency and reached safety. Your awareness and decision-making have been crucial in this successful escape.
    
    \n\nEND OF EXAMPLE\n\n
    Before responsing, make sure you tell what scenario you have selected to formulate a course based on the
    information provided and human input prompt. Make sure you select only one scenario and give your response
    to the exact format you have selected the scenario of.
    It is an absolute requirement to adhere to the tag words (words or tags enclosed in the Single quotation marks '').
    Give concise, relevant, clear, and descriptive instructions
    as you are a course creator that has expertise in molding asked information into one of the above four
    scenarios.
    Human: {human_input},{subject_name},Information relevant to human input:({input_documents}).Remember the information
    relevant to human input in no way overrides the format of a scenario. Use the information content and adhere to the
    format of the scenario you choose with all the tags, including [Media belonging to #]  tags enclosed in the single quotation marks ''.

    Chatbot:"""
)

prompt_linear = PromptTemplate(
    input_variables=["input_documents","human_input","subject_name"], #,"chat_history"],
    template="""
    You are an educational chat bot that helps in building training courses for human. 
    You prepare the courses adhering to the format of the Linear scenario given as example below, suitable to the type of information which
    is relevant to the human's input query or prompt.

    Make sure to give your response in the exact format for this Linear Scenario.
    It is an absolute requirement to adhere to the tag words (words or tags enclosed in the Single quotation marks '').
    Give concise, relevant, clear, and descriptive instructions
    as you are a course creator that has expertise in molding asked information into one of the above four
    scenarios.

    It is absolutely mandatory and required of you to adhere with the format and structure 
    of the below given Linear Scenario. Adhere with the tags or words in the single quotation marks ''
    because these tags are responsible for the structure of the scenario according to which you have 
    to formualte the course or linear scenario with human provided query and relevant information.  
    \n\nEXAMPLE FORMAT FOR EACH SCENARIO\n\n
    \nLINEAR SCENARIO:\n
    'LINEAR SCENARIO':
    'Learning Objectives':
    - Identify the basic steps in tying shoe laces.
    - Understand the importance of starting with even laces for effective tying.
    - Learn to create and manipulate loops to form a basic shoe lace knot.
    - Recognize different styles of laces and their tying techniques through media interaction.
    - Apply knowledge through quiz questions to reinforce learning of the shoe lace tying process.
    'Topic': How to Tie Shoe Laces
    [Media belonging to 'Topic', Image Description: An array of shoes with different styles of laces displayed on a white background. Overlay Tags: Tag 1 - 'Flat Laces', Tag 2 - 'Round Laces', Tag 3 - 'Colorful Laces'. Each tag, when clicked, leads to a short video showcasing the lacing technique for that style.]
    'Point 1': Ensure your shoe laces are even on both sides. Hold the ends of your laces, making sure they are of equal length.
    [Media belonging to 'Point 1', Video Description: A step-by-step guide on evening out your shoe laces before beginning the tying process. Overlay Tags: Tag 1 - 'Holding Technique', shows the correct way to hold laces. Tag 2 - 'Measuring Evenness', demonstrates how to ensure laces are even.]
    'Point 2': Cross the right lace over the left, pulling it underneath and through the loop formed, creating a simple knot.
    'Question 1 Point 2': What is the first action in tying shoe laces?
    Options: A. Make a bow. B. Cross the laces. C. Tie a double knot.
    Correct Answer: B. Cross the laces.
    'Point 3': Create a loop with each lace, holding them up to form what looks like two bunny ears.
    [Media belonging to 'Point 3', Image Description: Close-up of hands holding up two loops in the laces against a neutral background. Overlay Tags: Tag 1 - 'Loop Formation', detailed instructions on creating perfect loops. Tag 2 - 'Bunny Ear Technique', tips for maintaining loop shape.]
    'Point 4': Cross the two loops, pulling one loop through the space between them to form a secure knot.
    [Media belonging to 'Point 4', Video Description: Demonstrating the technique for crossing loops and securing the knot. Overlay Tags: Tag 1 - 'Crossing Loops', shows the crossing action. Tag 2 - 'Securing the Knot', provides additional tips for knot tightness.]
    'Quiz':
    'Question 1 Quiz Point 1': Which point involves checking if the laces are even?
    Options: A. Point 1. B. Point 2. C. Point 3.
    Correct Answer: A. Point 1.
    Score: 5 points
    'Question 2 Quiz Point 2': What shape do your hands make with the laces in Point 3?
    Options: A. Straight line. B. Bunny ears. C. Circle.
    Correct Answer: B. Bunny ears.
    Score: 5 points
    'Question 3 Quiz Point 4': How do you secure the knot?
    Options: A. By tying the loops together. B. By making another basic knot. C. No need to secure, it's already done.
    Correct Answer: A. By tying the loops together.
    Score: 5 points
    'Total Score: 15 points'
    \n\nEND OF EXAMPLE\n\n
    Human: {human_input},{subject_name},Information relevant to human input:({input_documents}). Use the information content 
    to mold the response that adheres to the format of this scenario with all the tags, including [Media belonging to #]  tags enclosed in the single quotation marks ''.
    Chatbot:"""
)

prompt_selfexploratory = PromptTemplate(
    input_variables=["input_documents","human_input","subject_name"], #,"chat_history"],
    template="""
    You are an educational chat bot that helps in building training courses for human. 
    You prepare the courses adhering to the format of the Self-Exploratory scenario given as example below, suitable to the type of information which
    is relevant to the human's input query or prompt.

    Make sure to give your response in the exact format for this Self-Exploratory Scenario.
    It is absolutely mandatory and required for you to adhere with the format and structure 
    of the below given Self-Exploratory Scenario. Adhere with the tags or words in the single quotation marks ''
    because these tags are responsible for the structure of the scenario according to which you have 
    to formualte the course or Self-Exploratory scenario with human provided query and relevant information.  
    \n\nEXAMPLE FORMAT FOR EACH SCENARIO\n\n
    \nSelf-Exploratory Scenario:\n
    'Self-Exploratory Scenario':
    'Learning Objectives':
    - Differentiate between various types of renewable energy sources, such as wind and solar energy.
    - Understand the basic mechanisms behind how wind turbines and solar panels generate electricity.
    - Recognize the environmental and societal benefits of transitioning to renewable energy sources.
    - Engage with interactive media to explore the technical and environmental aspects of renewable energies.
    - Apply critical thinking to assess the impact of renewable energy on reducing carbon footprint and greenhouse gas emissions.
    'Topic': Introduction to Renewable Energy
    'Scenario':
    The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.
    [Media belonging to 'Scenario', Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.]
    'Decision Point Topic': Choose a renewable energy source to explore how it works and its benefits.
    'Topic 1 'Decision Point Topic': Wind Energy
    'Topic 1 Step 1': Understanding how wind turbines convert wind into electricity.
    [Media belonging to 'Topic 1 Step 1': Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.]
    'Topic 1 Step 2': The environmental impact and benefits of wind energy.
    [Media belonging to 'Topic 1 Step 2': Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 1 Decision Point Topic)'?
    'Quiz Topic 1 Decision Point Topic':
    'Question 1 Quiz Topic 1 Decision Point Topic': What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points
    'Question 2 Quiz Topic 1 Decision Point Topic': True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points

    'Topic 2 Decision Point Topic': Solar Energy
    'Topic 2 Step 1': How solar panels convert sunlight into electrical energy.
    [Media belonging to 'Topic 2 Step 1': Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.]
    'Topic 2 Step 2': The role of solar energy in powering homes and businesses.
    [Media belonging to 'Topic 2 Step 2': Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.]
    'Question 1 Topic 2 Step 2': What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic
    'Topic 2 Step 3': Installation and maintenance of solar panel systems.
    [Media belonging to 'Topic 2 Step 3': Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.]
    'Topic 2 Step 4': Future innovations in solar technology.
    [Media belonging to 'Topic 2 Step 4': Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 2 Decision Point Topic)'?
    'Quiz Topic 2 Decision Point Topic':
    'Question 1 Quiz Topic 2 Decision Point Topic': Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points
    'Question 2 Quiz Topic 2 Decision Point Topic': True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points
    'Question 3 Quiz Topic 2 Decision Point Topic': Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points                                   
    \n\nEND OF EXAMPLE\n\n

    It is an absolute requirement to adhere to the tag words (words or tags enclosed in the Single quotation marks '').
    Give concise, relevant, clear, and descriptive instructions
    as you are a course creator that has expertise in molding asked information into one of the above four
    scenarios.
    Human: {human_input},{subject_name},Information relevant to human input:({input_documents}). Use the information content 
    to mold the response that adheres to the format of this scenario with all the tags, including [Media belonging to #]  tags enclosed in the single quotation marks ''.
    Chatbot:"""
)

prompt_simulation = PromptTemplate(
    input_variables=["input_documents","human_input","subject_name"], #,"chat_history"],
    template="""
    You are an educational chat bot that helps in building training courses for human. 
    You prepare the courses adhering to the format of the Simulation scenario given as example below, suitable to the type of information which
    is relevant to the human's input query or prompt.

    Make sure to give your response in the exact format for this Simulation Scenario.
    It is absolutely mandatory and required for you to adhere with the format and structure 
    of the below given Simulation Scenario. Adhere with the tags or words in the single quotation marks ''
    because these tags are responsible for the structure of the scenario according to which you have 
    to formualte the course or Simulation scenario with human provided query and relevant information.  
    \nSIMULATION SCENARIO:\n
    'SIMULATION SCENARIO':
    'Learning Objectives':
    - Navigate through a corporate archive section to understand its layout and the function of different sub-sections.
    - Identify the operational processes within the printing office, including the use and benefits of specific equipment like scanners and printers.
    - Explore the management and correspondence functions within the head office, including team organization and documentation.
    - Investigate the library section to learn how publications are stored, accessed, and managed digitally and physically.
    - Apply decision-making skills to explore detailed aspects of archive management, such as digital storage solutions and publication cataloging.
    'Topic': Introduction to the Archive Section
    'Scenario': You are in the main lounge of a company building on your first day for an office tour. You are being shown an Archive Section where there different sub-sections situated at the left, right and upstairs from the main lounge.
    [Media belonging to 'Scenario', Description: A 360-degree image of a corporate lounge with paths leading to the left, right, and upstairs. Overlay Tags: Tag 1 - 'Left Path': An arrow pointing left with a caption: "To the Printing Office". Tag 2 - 'Right Path': An arrow pointing right with a caption: "To the Library". Tag 3 - 'Upstairs': An arrow pointing upwards with a caption: "To the Head Office".]
    'Decision Point Main': There are 3 paths to choose from the main lounge. Left, Right and Upstairs.
    'Timer Decision Point Main': 1 minutes
    'Choice 1 Decision Point Main': Left path.
    'Choice 1 Consequence Decision Point Main': You reach the printing office of Archive Section when you move left from the lounge. 'Move on to Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point Main', Description: Picture of an office with various printers and a large dedicated scanner. Overlay Tags: Tag 1 - 'Printer': Close-up on a printer with a caption: "60 Copies per Minute". Tag 2 - 'Dedicated Scanner': Close-up on a scanner with a caption: "Scans up to A0 size paper".]
    'Choice 2 Decision Point Main': Upstairs.
    'Choice 2 Consequence Decision Point Main': You reach to the head office of Archive Section where management and correspondence is carried out. 'Move on to Decision Point 2'.
    [Media belonging to 'Choice 2 Consequence Decision Point Main', Description: A picture showing an office that may belong to a manager, with documents and a computer on a desk. Overlay Tags: Tag 1 - 'Manager's Desk': A close-up on the desk with a caption: "Team State Document".]
    'Choice 3 Decision Point Main': Right path.
    'Choice 3 Consequence Decision Point Main': You reach the library of the Archive Section. 'Move on to Decision Point 3'.
    [Media belonging to 'Choice 3 Consequence Decision Point Main', Description: Picture of a library with shelves full of books and a computer setup for accessing records. Overlay Tags: Tag 1 - 'Computer Records': Zoom in on the computer screen with a caption: "Access Publication Records Here". Tag 2 - 'Soft Storage Cabinets': Focus on closed cabinets with a caption: "Contains Digital Archives".]

    'Decision Point 1': The printing office is used to print the publications. They are given to the library subsection after printing. There are 2 machines in the printing office, a large dedicated scanner and a large printer.
    'Timer Decision Point 1': 5 minutes
    'Choice 1 Decision Point 1': Dedicated Scanner.
    'Choice 1 Consequence Decision Point 1': The dedicated scanner can scan upto A0 size paper, mostly used for scanning large engineering blue-prints. 'Move on to Result Choice 1 Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point 1', Description: Video of a large drawing being scanned by a scanner. Overlay Tags: Tag 1 - 'Scanning Process': Detailed video showing how the dedicated scanner operates, including paper feeding, scanning mechanism, and output. Tag 2 - 'Blueprint Scanning': Tips on preparing and positioning large engineering blueprints for scanning.]
    'Result Choice 1 Decision Point 1': Score 5
    'Choice 2 Decision Point 1': Printer. 
    'Choice 2 Consequence Decision Point 1': This printer can print 60 Copies per Minute. 'Move on to Result Choice 2 Decision Point 1'.
    [Media belonging to 'Choice 2 Consequence Decision Point 1', Description: Video of a large printer in operation, showing rapid printing. Overlay Tags: Tag 1 - 'Printer Capabilities': Explanation of printer features, including speed, paper handling, and quality settings. Tag 2 - 'High-Volume Printing': Best practices for managing large print jobs efficiently.]
    'Result Choice 2 Decision Point 1': Score 5

    'Decision Point 2': The head office is used for correspondence, and management.
    'Timer Decision Point 2': 5 minutes
    'Choice 1 Decision Point 2': See the team state document.
    'Choice 1 Consequence Decision Point 2': The team state document shows all the employees and their data with work responsibilities and leave records. 'Move on to Result Choice 1 Decision Point 2'.
    [Media belonging to 'Choice 1 Consequence Decision Point 2', Description: Picture of an office desk with various documents and a computer screen displaying an employee database. Overlay Tags: Tag 1 - 'Employee Database': Detailed view of the database interface with functionalities highlighted. Tag 2 - 'Document Overview': Close-up on key documents that manage team responsibilities and leave schedules.]
    'Result Choice 1 Decision Point 2': Score 10

    'Decision Point 3': The library is used to keep all the record of publications printed in the printing section.
    'Timer Decision Point 3': 10 minutes
    'Choice 1 Decision Point 3': View computer records of publications.
    'Choice 1 Consequence Decision Point 3': The computer records contains record of available publications and their locations on the shelves. 'Move on to Result Choice 1 Decision Point 3'.
    [Media belonging to 'Choice 1 Consequence Decision Point 3', Description: An image of a computer screen displaying a digital catalog of publications with search functionality and shelf location information. Overlay Tags: Tag 1 - 'Digital Catalog': Interactive demo on navigating the digital records to find publications. Tag 2 - 'Locating Publications': Instructions on how to use the catalog to find the exact shelf location of a book or document.]
    'Result Choice 1 Decision Point 3': Score 5
    'Choice 2 Decision Point 3': View the closed soft storage cabinets.
    'Choice 2 Consequence Decision Point 3': The cabinet is opened for you to see inside. 'Move on to Decision Point 4'.
    [Media belonging to 'Choice 2 Consequence Decision Point 3', Description: Picture of an opened storage cabinet revealing various labeled compartments and digital storage media. Overlay Tags: Tag 1 - 'Storage Organization': Overview of how the cabinet is organized for efficient storage and retrieval. Tag 2 - 'Digital Media Storage': Insights into the preservation and categorization of digital archives.]

    'Decision Point 4': The cabinets contains various disks that stores the content of the publications in soft form, to be given with the publications at the time of issue to customers.
    'Timer Decision Point 4': 3 minutes
    'Choice 1 Decision Point 4': Inspect a disk in the cabinet.
    'Choice 1 Consequence Decision Point 4': On the disk, information is given about what publication this disk belongs to and where the location of that publication in the shelves is. 'Move on to Result Choice 1 Decision Point 4'.
    [Media belonging to 'Choice 1 Consequence Decision Point 4', Description: A close-up of a disk with labels indicating the publication it accompanies and a map showing its shelf location in the library. Overlay Tags: Tag 1 - 'Disk Content Overview': A quick guide on the information provided on the disk, including the publication it belongs to. Tag 2 - 'Finding Publications': Instructions on using the disk's label to locate the physical publication in the library.]
    'Result Choice 1 Decision Point 4': Score 20
    \n\nEND OF EXAMPLE\n\n

    It is an absolute requirement to adhere to the tag words (words or tags enclosed in the Single quotation marks '').
    Give concise, relevant, clear, and descriptive instructions
    as you are a course creator that has expertise in molding asked information into one of the above four
    scenarios.
    Human: {human_input},{subject_name},Information relevant to human input:({input_documents}). Use the information content 
    to mold the response that adheres to the format of this scenario with all the tags, including [Media belonging to #]  tags enclosed in the single quotation marks ''.
    Chatbot:"""
)

prompt_escaperoom = PromptTemplate(
    input_variables=["input_documents","human_input","subject_name"], #,"chat_history"],
    template="""
    You are an educational chat bot that helps in building training courses for human. 
    You prepare the courses adhering to the format of the Escape Room scenario given as example below, suitable to the type of information which
    is relevant to the human's input query or prompt.

    Make sure to give your response in the exact format for this Escape Room Scenario.
    It is absolutely mandatory and required for you to adhere with the format and structure 
    of the below given Escape Room Scenario. Adhere with the tags or words in the single quotation marks ''
    because these tags are responsible for the structure of the scenario according to which you have 
    to formualte the course or Escape Room scenario with human provided query and relevant information.  
    \nESCAPE ROOM SCENARIO:\n
    'ESCAPE ROOM SCENARIO':
    'Learning Objectives':
    - Recognize the importance of quick and informed decision-making during a fire emergency.
    - Identify fire safety protocols, including the avoidance of elevators and the use of staircases for evacuation.
    - Implement strategies to minimize smoke inhalation, such as covering your nose and mouth and staying low to the ground.
    - Evaluate the safest exit routes, understanding the difference between familiar exits and potentially dangerous shortcuts.
    - Practice locating and moving to designated assembly points post-evacuation for accountability and further instructions from emergency services.
    'Topic': Exiting the Building in a Fire Emergency
    'Introduction': You're in a multi-story building when an alarm sounds, signaling a fire emergency. Smoke is starting to fill the corridors, and it's imperative to leave the building as quickly and safely as possible. Your actions and decisions will determine your fate.
    [Media belonging to 'Introduction': Description: A 360-degree image of a smoke-filled corridor with emergency lights flashing. Overlay Tags: Tag 1 - 'Emergency Lighting': An arrow or glow around the emergency lights to guide the way. Tag 2 - 'Smoke Density': Visual cues on the smoke's density, suggesting lower areas have less smoke.]

    'Decision Point 1': Identify the safest initial route to exit the building.
    'Clue 1 Decision Point 1': Fire safety protocols suggest avoiding elevators during a fire.
    'Clue 2 Decision Point 1': Look for illuminated exit signs that indicate the pathway to safety.
    'Timer Decision Point 1': 2 minutes
    'Correct Choice Decision Point 1': Head towards the nearest staircase.
    'Correct Choice Consequence Decision Point 1': You find the staircase and start descending safely. 'Move on to Decision Point 2'.
    [Media belonging to 'Correct Choice Consequence Decision Point 1': Description: Picture of an illuminated 'Exit' sign above a stairwell. Overlay Tags: Tag 1 - 'Staircase Access': An arrow or circle highlighting the exit sign and the entrance to the staircase. Tag 2 - 'Fire Safety Tip': A note on why stairs are safer than elevators during a fire.]
    'Incorrect Choice Decision Point 1': Attempt to use the elevator.
    'Incorrect Choice Consequence Decision Point 1': The elevator is non-operational during the emergency, wasting precious time. 'Jump back to Decision Point 1'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 1': Description: Video clip showing an "Out of Service" message on the elevator panel. Overlay Tags: Tag 1 - 'Out of Service Notice': Close-up on the elevator panel showing the message. Tag 2 - 'Reminder': A note reminding that elevators should not be used during fires.]

    'Decision Point 2': Encounter smoke in the staircase. Choose how to proceed.
    'Clue 1 Decision Point 2': Smoke rises, so lower air might be clearer.
    'Clue 2 Decision Point 2': Covering your nose and mouth can help filter harmful particles.
    'Timer Decision Point 2': 2 minutes
    'Correct Choice Decision Point 2': Use a cloth to cover your nose and mouth, and proceed down the stairs, staying as low as possible.
    'Correct Choice Consequence Decision Point 2': You manage to breathe more easily and continue your descent. 'Move on to Decision Point 3'.

    [Media belonging to 'Correct Choice Consequence Decision Point 2': Description: Picture of a person using a cloth over their mouth, crouched low on stairs. Overlay Tags: Tag 1 - 'Proper Cloth Use': Instructions for covering nose and mouth effectively. Tag 2 - 'Stay Low Strategy': Demonstration of the low posture to stay below the smoke layer.]
    'Incorrect Choice Decision Point 2': Ignore the smoke and run through it as quickly as possible.
    'Incorrect Choice Consequence Decision Point 2': You cough and struggle to see, slowing your progress significantly. 'Jump back to Decision Point 2'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 2': Description: Video clip of a person coughing and struggling to navigate through smoke. Overlay Tags: Tag 1 - 'Effects of Smoke Inhalation': Visual on the impact of smoke on breathing and vision. Tag 2 - 'Warning': Advice against rushing through heavy smoke.]

    'Decision Point 3': Choose between two exits: a nearby door leading outside or continue down to the main exit.
    'Clue 1 for Decision Point 3': Not all doors lead to safe exits; some may lead to areas that are more dangerous.
    'Clue 2 for Decision Point 3': Familiar exits are generally safer but might be more crowded.
    'Timer Decision Point 3': 3 minutes
    'Correct Choice Decision Point 3': Continue down to the main exit you're familiar with.
    'Correct Choice Consequence Decision Point 3': Despite the crowd, you exit safely into the open air. 'Move on to Decision Point 4'.
    [Media belonging to 'Correct Choice Consequence Decision Point 3': Description: 360-degree image of the crowded but orderly evacuation at the main building exit. Overlay Tags: Tag 1 - 'Crowded Exit Strategy': Tips on navigating through a crowded exit safely. Tag 2 - 'Familiar Path': Highlighting the familiar route, reinforcing the choice for safety.]
    'Incorrect Choice Decision Point 3': Try the nearby door for a quicker escape.
    'Incorrect Choice Consequence Decision Point 3': The door leads to a dead-end, forcing you to turn back and lose time. 'Jump back to Decision Point 3'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 3': Description: Picture of a dead-end area with no exit, smoke building up. Overlay Tags: Tag 1 - 'Dead-End Warning': A caution symbol or text indicating the dead-end. Tag 2 - 'Smoke Accumulation': Visual cues on the increasing smoke, emphasizing the urgency to find the correct exit.]

    'Decision Point 4': After exiting, you must choose where to go next.
    'Clue 1 for Decision Point 4': Emergency services recommend moving away from the building to allow access for firefighters and ambulances.
    'Clue 2 for Decision Point 4': Designated assembly points are set up to account for everyone.
    'Timer Decision Point 4': 2 minutes
    'Correct Choice Decision Point 4': Head to the designated assembly point.
    'Correct Choice Consequence Decision Point 4': You are accounted for and receive instructions from emergency personnel. 'Move on to Escape Block'.
    [Media belonging to 'Correct Choice Consequence Decision Point 4': Description: Picture of survivors gathering at the designated assembly point with emergency services in attendance. Overlay Tags: Tag 1 - 'Assembly Point Location': Arrows or markers pointing to the assembly area. Tag 2 - 'Emergency Instructions': A depiction of emergency personnel providing guidance to evacuees.]
    'Incorrect Choice Decision Point 4': Stay close to the building to watch what happens.
    'Incorrect Choice Consequence Decision Point 4': Emergency personnel usher you away for your safety, delaying their work. 'Jump back to Decision Point 4'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 4': Description: Video clip of emergency personnel directing people away from the building. Overlay Tags: Tag 1 - 'Safety Direction': Visual of emergency personnel guiding evacuees to safety. Tag 2 - 'Hazard Area': Highlighting the danger zone to avoid near the building.]

    'Escape Block': Congratulations, you've navigated through the fire emergency and reached safety. Your awareness and decision-making have been crucial in this successful escape.
    \n\nEND OF EXAMPLE\n\n

    It is an absolute requirement to adhere to the tag words (words or tags enclosed in the Single quotation marks '').
    Give concise, relevant, clear, and descriptive instructions
    as you are a course creator that has expertise in molding asked information into one of the above four
    scenarios.
    Human: {human_input},{subject_name},Information relevant to human input:({input_documents}). Use the information content 
    to mold the response that adheres to the format of this scenario with all the tags, including [Media belonging to #]  tags enclosed in the single quotation marks ''.
    Chatbot:"""
)
# chain = load_qa_chain(
#     llm=llm, chain_type="stuff", prompt=prompt
# )
memory_LLM = ConversationBufferWindowMemory(k=2)

def TALK_WITH_RAG(query, docsearch, llm,scenario,memory):
    print("TALK_WITH_RAG Initiated!")
    docs = docsearch.similarity_search(query, k=3)
    docs_main = " ".join([d.page_content for d in docs])
    # chain = load_qa_chain(
    #     llm=llm, chain_type="stuff", prompt=prompt
    # )
    user = memory[-1].get('user', "") if memory else ""
    print("LCD USER",user)
    bot = memory[-1].get('bot', "") if memory else ""
    print("LCD BOT",bot)
    memory_LLM.save_context({"input": user}, {"output": bot})
    print(memory_LLM.load_memory_variables({}))
    if scenario == 1:
        chain = LLMChain(prompt=prompt_linear, llm=llm,memory=memory_LLM)
        print("SCENARIO ====prompt_linear",scenario)
    elif scenario == 2:
        chain = LLMChain(prompt=prompt_selfexploratory, llm=llm,memory=memory_LLM)
        print("SCENARIO ====prompt_selfexploratory",scenario)
    elif scenario == 3:
        chain = LLMChain(prompt=prompt_simulation, llm=llm,memory=memory_LLM)
        print("SCENARIO ====prompt_simulation",scenario)
    elif scenario == 4:
        chain = LLMChain(prompt=prompt_escaperoom, llm=llm,memory=memory_LLM)
        print("SCENARIO ====prompt_escaperoom",scenario)
    elif scenario == 0:
        print("SCENARIO ====PROMPT",scenario)
        chain = LLMChain(prompt=prompt, llm=llm,memory=memory_LLM)
    
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
    print(docs_main)
    return chain, docs_main, query, subject_name


def GENERATE_GRAPHML(bot_last_reply,llmsx):
    print("This is last reply",bot_last_reply)

    GRAPHML_PROMPT_ESCAPE_ROOM = PromptTemplate(input_variables=['text'], template="""You are a networked intelligence helping a human track knowledge by giving providing them with
    graphml having various nodes and edges representing information about all
    relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    Extract all of the nodes and edges from the text. The nodes are connected to relevant other nodes by edges. These edges have also information in them.
    The concept is like a knowledge triplet, which is a clause that contains a subject, a predicate, and an object.
    The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.
    You are only allowed to form the node ids of words or tags enclosed in the single quotation marks ' '.
    \n\nEXAMPLE\n\n
    \nESCAPE ROOM SCENARIO:\n
    'Learning Objectives':
    - Recognize the importance of quick and informed decision-making during a fire emergency.
    - Identify fire safety protocols, including the avoidance of elevators and the use of staircases for evacuation.
    - Implement strategies to minimize smoke inhalation, such as covering your nose and mouth and staying low to the ground.
    - Evaluate the safest exit routes, understanding the difference between familiar exits and potentially dangerous shortcuts.
    - Practice locating and moving to designated assembly points post-evacuation for accountability and further instructions from emergency services.
    'Topic': Exiting the Building in a Fire Emergency
    'Introduction': You're in a multi-story building when an alarm sounds, signaling a fire emergency. Smoke is starting to fill the corridors, and it's imperative to leave the building as quickly and safely as possible. Your actions and decisions will determine your fate.
    [Media belonging to 'Introduction': Description: A 360-degree image of a smoke-filled corridor with emergency lights flashing. Overlay Tags: Tag 1 - 'Emergency Lighting': An arrow or glow around the emergency lights to guide the way. Tag 2 - 'Smoke Density': Visual cues on the smoke's density, suggesting lower areas have less smoke.]

    'Decision Point 1': Identify the safest initial route to exit the building.
    'Clue 1 Decision Point 1': Fire safety protocols suggest avoiding elevators during a fire.
    'Clue 2 Decision Point 1': Look for illuminated exit signs that indicate the pathway to safety.
    'Timer Decision Point 1': 2 minutes
    'Correct Choice Decision Point 1': Head towards the nearest staircase.
    'Correct Choice Consequence Decision Point 1': You find the staircase and start descending safely. 'Move on to Decision Point 2'.
    [Media belonging to 'Correct Choice Consequence Decision Point 1': Description: Picture of an illuminated 'Exit' sign above a stairwell. Overlay Tags: Tag 1 - 'Staircase Access': An arrow or circle highlighting the exit sign and the entrance to the staircase. Tag 2 - 'Fire Safety Tip': A note on why stairs are safer than elevators during a fire.]
    'Incorrect Choice Decision Point 1': Attempt to use the elevator.
    'Incorrect Choice Consequence Decision Point 1': The elevator is non-operational during the emergency, wasting precious time. 'Jump back to Decision Point 1'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 1': Description: Video clip showing an "Out of Service" message on the elevator panel. Overlay Tags: Tag 1 - 'Out of Service Notice': Close-up on the elevator panel showing the message. Tag 2 - 'Reminder': A note reminding that elevators should not be used during fires.]

    'Decision Point 2': Encounter smoke in the staircase. Choose how to proceed.
    'Clue 1 Decision Point 2': Smoke rises, so lower air might be clearer.
    'Clue 2 Decision Point 2': Covering your nose and mouth can help filter harmful particles.
    'Timer Decision Point 2': 2 minutes
    'Correct Choice Decision Point 2': Use a cloth to cover your nose and mouth, and proceed down the stairs, staying as low as possible.
    'Correct Choice Consequence Decision Point 2': You manage to breathe more easily and continue your descent. 'Move on to Decision Point 3'.

    [Media belonging to 'Correct Choice Consequence Decision Point 2': Description: Picture of a person using a cloth over their mouth, crouched low on stairs. Overlay Tags: Tag 1 - 'Proper Cloth Use': Instructions for covering nose and mouth effectively. Tag 2 - 'Stay Low Strategy': Demonstration of the low posture to stay below the smoke layer.]
    'Incorrect Choice Decision Point 2': Ignore the smoke and run through it as quickly as possible.
    'Incorrect Choice Consequence Decision Point 2': You cough and struggle to see, slowing your progress significantly. 'Jump back to Decision Point 2'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 2': Description: Video clip of a person coughing and struggling to navigate through smoke. Overlay Tags: Tag 1 - 'Effects of Smoke Inhalation': Visual on the impact of smoke on breathing and vision. Tag 2 - 'Warning': Advice against rushing through heavy smoke.]

    'Decision Point 3': Choose between two exits: a nearby door leading outside or continue down to the main exit.
    'Clue 1 for Decision Point 3': Not all doors lead to safe exits; some may lead to areas that are more dangerous.
    'Clue 2 for Decision Point 3': Familiar exits are generally safer but might be more crowded.
    'Timer Decision Point 3': 3 minutes
    'Correct Choice Decision Point 3': Continue down to the main exit you're familiar with.
    'Correct Choice Consequence Decision Point 3': Despite the crowd, you exit safely into the open air. 'Move on to Decision Point 4'.
    [Media belonging to 'Correct Choice Consequence Decision Point 3': Description: 360-degree image of the crowded but orderly evacuation at the main building exit. Overlay Tags: Tag 1 - 'Crowded Exit Strategy': Tips on navigating through a crowded exit safely. Tag 2 - 'Familiar Path': Highlighting the familiar route, reinforcing the choice for safety.]
    'Incorrect Choice Decision Point 3': Try the nearby door for a quicker escape.
    'Incorrect Choice Consequence Decision Point 3': The door leads to a dead-end, forcing you to turn back and lose time. 'Jump back to Decision Point 3'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 3': Description: Picture of a dead-end area with no exit, smoke building up. Overlay Tags: Tag 1 - 'Dead-End Warning': A caution symbol or text indicating the dead-end. Tag 2 - 'Smoke Accumulation': Visual cues on the increasing smoke, emphasizing the urgency to find the correct exit.]

    'Decision Point 4': After exiting, you must choose where to go next.
    'Clue 1 for Decision Point 4': Emergency services recommend moving away from the building to allow access for firefighters and ambulances.
    'Clue 2 for Decision Point 4': Designated assembly points are set up to account for everyone.
    'Timer Decision Point 4': 2 minutes
    'Correct Choice Decision Point 4': Head to the designated assembly point.
    'Correct Choice Consequence Decision Point 4': You are accounted for and receive instructions from emergency personnel. 'Move on to Escape Block'.
    [Media belonging to 'Correct Choice Consequence Decision Point 4': Description: Picture of survivors gathering at the designated assembly point with emergency services in attendance. Overlay Tags: Tag 1 - 'Assembly Point Location': Arrows or markers pointing to the assembly area. Tag 2 - 'Emergency Instructions': A depiction of emergency personnel providing guidance to evacuees.]
    'Incorrect Choice Decision Point 4': Stay close to the building to watch what happens.
    'Incorrect Choice Consequence Decision Point 4': Emergency personnel usher you away for your safety, delaying their work. 'Jump back to Decision Point 4'.
    [Media belonging to 'Incorrect Choice Consequence Decision Point 4': Description: Video clip of emergency personnel directing people away from the building. Overlay Tags: Tag 1 - 'Safety Direction': Visual of emergency personnel guiding evacuees to safety. Tag 2 - 'Hazard Area': Highlighting the danger zone to avoid near the building.]

    'Escape Block': Congratulations, you've navigated through the fire emergency and reached safety. Your awareness and decision-making have been crucial in this successful escape.
                                  
    Output:\n
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d0" for="edge" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="undirected">
        <node id="Learning Objectives"/>
        <node id="Topic"/>
        <node id="Introduction"/>
        <node id="Media belonging to 'Introduction'"/>
        <node id="Decision Point 1"/>
        <node id="Clue 1 Decision Point 1"/>
        <node id="Clue 2 Decision Point 1"/>
        <node id="Timer Decision Point 1"/>
        <node id="Correct Choice Decision Point 1"/>
        <node id="Correct Choice Consequence Decision Point 1"/>
        <node id="Media belonging to 'Correct Choice Consequence Decision Point 1'"/>
        <node id="Incorrect Choice Decision Point 1"/>
        <node id="Incorrect Choice Consequence Decision Point 1"/>
        <node id="Media belonging to 'Incorrect Choice Consequence Decision Point 1'"/>
        <node id="Decision Point 2"/>
        <node id="Clue 1 Decision Point 2"/>
        <node id="Clue 2 Decision Point 2"/>
        <node id="Timer Decision Point 2"/>
        <node id="Correct Choice Decision Point 2"/>
        <node id="Correct Choice Consequence Decision Point 2"/>
        <node id="Media belonging to 'Correct Choice Consequence Decision Point 2'"/>
        <node id="Incorrect Choice Decision Point 2"/>
        <node id="Incorrect Choice Consequence Decision Point 2"/>
        <node id="Media belonging to 'Incorrect Choice Consequence Decision Point 2'"/>
        <node id="Decision Point 3"/>
        <node id="Clue 1 Decision Point 3"/>
        <node id="Clue 2 Decision Point 3"/>
        <node id="Timer Decision Point 3"/>
        <node id="Correct Choice Decision Point 3"/>
        <node id="Correct Choice Consequence Decision Point 3"/>
        <node id="Media belonging to 'Correct Choice Consequence Decision Point 3'"/>
        <node id="Incorrect Choice Decision Point 3"/>
        <node id="Incorrect Choice Consequence Decision Point 3"/>
        <node id="Media belonging to 'Incorrect Choice Consequence Decision Point 3'"/>
        <node id="Decision Point 4"/>
        <node id="Clue 1 Decision Point 4"/>
        <node id="Clue 2 Decision Point 4"/>
        <node id="Timer Decision Point 4"/>
        <node id="Correct Choice Decision Point 4"/>
        <node id="Correct Choice Consequence Decision Point 4"/>
        <node id="Media belonging to 'Correct Choice Consequence Decision Point 4'"/>
        <node id="Incorrect Choice Decision Point 4"/>
        <node id="Incorrect Choice Consequence Decision Point 4"/>
        <node id="Media belonging to 'Incorrect Choice Consequence Decision Point 4'"/>
        <node id="Escape Block"/>
        <edge source="Learning Objectives" target="Topic">
        <data key="d0">- Recognize the importance of quick and informed decision-making during a fire emergency.
        - Identify fire safety protocols, including the avoidance of elevators and the use of staircases for evacuation.
        - Implement strategies to minimize smoke inhalation, such as covering your nose and mouth and staying low to the ground.
        - Evaluate the safest exit routes, understanding the difference between familiar exits and potentially dangerous shortcuts.
        - Practice locating and moving to designated assembly points post-evacuation for accountability and further instructions from emergency services.
        </data>
        </edge>                                                           
        <edge source="Topic" target="Introduction">
        <data key="d0">Exiting the Building in a Fire Emergency</data>
        </edge>
        <edge source="Media belonging to 'Introduction'" target="Introduction">
        <data key="d0">Description: A 360-degree image of a smoke-filled corridor with emergency lights flashing. Overlay Tags: Tag 1 - 'Emergency Lighting': An arrow or glow around the emergency lights to guide the way. Tag 2 - 'Smoke Density': Visual cues on the smoke's density, suggesting lower areas have less smoke.</data>
        </edge>
        <edge source="Introduction" target="Decision Point 1">
        <data key="d0">You're in a multi-story building when an alarm sounds, signaling a fire emergency. Smoke is starting to fill the corridors, and it's imperative to leave the building as quickly and safely as possible. Your actions and decisions will determine your fate.</data>
        </edge>
        <edge source=" Decision Point 1" target="Decision Point 1">
        <data key="d0">Identify the safest initial route to exit the building.</data>
        </edge>
        <edge source="Decision Point 1" target="Clue 1 Decision Point 1">
        <data key="d0">Fire safety protocols suggest avoiding elevators during a fire</data>
        </edge>
        <edge source="Decision Point 1" target="Clue 2 Decision Point 1">
        <data key="d0">Look for illuminated exit signs that indicate the pathway to safety</data>
        </edge>
        <edge source="Decision Point 1" target="Timer Decision Point 1">
        <data key="d0">2 minutes</data>
        </edge>
        <edge source="Decision Point 1" target="Correct Choice Decision Point 1">
        <data key="d0">Head towards the nearest staircase</data>
        </edge>
        <edge source="Correct Choice Decision Point 1" target="Correct Choice Consequence Decision Point 1">
        <data key="d0">You find the staircase and start descending safely.</data>
        </edge>
        <edge source=" Correct Choice Consequence Decision Point 1" target="Decision Point 2">
        <data key="d0">Move on to Decision Point 2</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 1" target="Media belonging to 'Correct Choice Consequence Decision Point 1'">
        <data key="d0">Description: Picture of an illuminated 'Exit' sign above a stairwell. Overlay Tags: Tag 1 - 'Staircase Access': An arrow or circle highlighting the exit sign and the entrance to the staircase. Tag 2 - 'Fire Safety Tip': A note on why stairs are safer than elevators during a fire.</data>
        </edge>
        <edge source="Decision Point 1" target="Incorrect Choice Decision Point 1">
        <data key="d0">Attempt to use the elevator</data>
        </edge>
        <edge source="Incorrect Choice Decision Point 1" target="Incorrect Choice Consequence Decision Point 1">
        <data key="d0">The elevator is non-operational during the emergency, wasting precious time</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 1" target="Decision Point 1">
        <data key="d0">Jump back to Decision Point 1</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 1" target="Media belonging to 'Incorrect Choice Consequence Decision Point 1'">
        <data key="d0">Description: Video clip showing an "Out of Service" message on the elevator panel. Overlay Tags: Tag 1 - 'Out of Service Notice': Close-up on the elevator panel showing the message. Tag 2 - 'Reminder': A note reminding that elevators should not be used during fires.</data>
        </edge>
        <edge source=" Decision Point 2" target="Decision Point 2">
        <data key="d0">Encounter smoke in the staircase. Choose how to proceed.</data>
        </edge>
        <edge source="Decision Point 2" target="Clue 1 Decision Point 2">
        <data key="d0">Smoke rises, so lower air might be clearer.</data>
        </edge>
        <edge source="Decision Point 2" target="Clue 2 Decision Point 2">
        <data key="d0">Covering your nose and mouth can help filter harmful particles.</data>
        </edge>
        <edge source="Decision Point 2" target="Timer Decision Point 2">
        <data key="d0">2 minutes</data>
        </edge>
        <edge source="Decision Point 2" target="Correct Choice Decision Point 2">
        <data key="d0">Use a cloth to cover your nose and mouth, and proceed down the stairs, staying as low as possible.</data>
        </edge>
        <edge source="Correct Choice Decision Point 2" target="Correct Choice Consequence Decision Point 2">
        <data key="d0">You manage to breathe more easily and continue your descent.</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 2" target="Decision Point 3">
        <data key="d0">Move on to Decision Point 3</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 2" target="Media belonging to 'Correct Choice Consequence Decision Point 2'">
        <data key="d0">Description: Picture of a person using a cloth over their mouth, crouched low on stairs. Overlay Tags: Tag 1 - 'Proper Cloth Use': Instructions for covering nose and mouth effectively. Tag 2 - 'Stay Low Strategy': Demonstration of the low posture to stay below the smoke layer.</data>
        </edge>
        <edge source="Decision Point 2" target="Incorrect Choice Decision Point 2">
        <data key="d0">Ignore the smoke and run through it as quickly as possible.</data>
        </edge>
        <edge source="Incorrect Choice Decision Point 2" target="Incorrect Choice Consequence Decision Point 2">
        <data key="d0">You cough and struggle to see, slowing your progress significantly.</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 2" target="Decision Point 2">
        <data key="d0">Jump back to Decision Point 2</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 2" target="Media belonging to 'Incorrect Choice Consequence Decision Point 2'">
        <data key="d0">Description: Video clip of a person coughing and struggling to navigate through smoke. Overlay Tags: Tag 1 - 'Effects of Smoke Inhalation': Visual on the impact of smoke on breathing and vision. Tag 2 - 'Warning': Advice against rushing through heavy smoke.</data>
        </edge>
        <edge source="Decision Point 3" target="Decision Point 3">
        <data key="d0">Choose between two exits: a nearby door leading outside or continue down to the main exit.</data>
        </edge>
        <edge source="Decision Point 3" target="Clue 1 Decision Point 3">
        <data key="d0">Not all doors lead to safe exits; some may lead to areas that are more dangerous.</data>
        </edge>
        <edge source="Decision Point 3" target="Clue 2 Decision Point 3">
        <data key="d0">Familiar exits are generally safer but might be more crowded.</data>
        </edge>
        <edge source="Decision Point 3" target="Timer Decision Point 3">
        <data key="d0">3 minutes</data>
        </edge>
        <edge source="Decision Point 3" target="Correct Choice Decision Point 3">
        <data key="d0">Continue down to the main exit you're familiar with.</data>
        </edge>
        <edge source="Correct Choice Decision Point 3" target="Correct Choice Consequence Decision Point 3">
        <data key="d0">Despite the crowd, you exit safely into the open air.</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 3" target="Decision Point 4">
        <data key="d0">Move on to Decision Point 4</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 3" target="Media belonging to 'Correct Choice Consequence Decision Point 3'">
        <data key="d0">Description: 360-degree image of the crowded but orderly evacuation at the main building exit. Overlay Tags: Tag 1 - 'Crowded Exit Strategy': Tips on navigating through a crowded exit safely. Tag 2 - 'Familiar Path': Highlighting the familiar route, reinforcing the choice for safety.</data>
        </edge>
        <edge source="Decision Point 3" target="Incorrect Choice Decision Point 3">
        <data key="d0">Try the nearby door for a quicker escape.</data>
        </edge>
        <edge source="Incorrect Choice Decision Point 3" target="Incorrect Choice Consequence Decision Point 3">
        <data key="d0">The door leads to a dead-end, forcing you to turn back and lose time.</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 3" target="Decision Point 3">
        <data key="d0">Jump back to Decision Point 3</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 3" target="Media belonging to 'Incorrect Choice Consequence Decision Point 3'">
        <data key="d0">Description: Picture of a dead-end area with no exit, smoke building up. Overlay Tags: Tag 1 - 'Dead-End Warning': A caution symbol or text indicating the dead-end. Tag 2 - 'Smoke Accumulation': Visual cues on the increasing smoke, emphasizing the urgency to find the correct exit.</data>
        </edge>
        <edge source="Decision Point 4" target="Decision Point 4">
        <data key="d0">After exiting, you must choose where to go next.</data>
        </edge>
        <edge source="Decision Point 4" target="Clue 1 Decision Point 4">
        <data key="d0">Emergency services recommend moving away from the building to allow access for firefighters and ambulances.</data>
        </edge>
        <edge source="Decision Point 4" target="Clue 2 Decision Point 4">
        <data key="d0">Designated assembly points are set up to account for everyone.</data>
        </edge>
        <edge source="Decision Point 4" target="Timer Decision Point 4">
        <data key="d0">2 minutes</data>
        </edge>
        <edge source="Decision Point 4" target="Correct Choice Decision Point 4">
        <data key="d0">Head to the designated assembly point.</data>
        </edge>
        <edge source="Correct Choice Decision Point 4" target="Correct Choice Consequence Decision Point 4">
        <data key="d0">You are accounted for and receive instructions from emergency personnel.</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 4" target="Escape Block">
        <data key="d0">Move on to Escape Block</data>
        </edge>
        <edge source="Correct Choice Consequence Decision Point 4" target="Media belonging to 'Correct Choice Consequence Decision Point 4'">
        <data key="d0">Description: Picture of survivors gathering at the designated assembly point with emergency services in attendance. Overlay Tags: Tag 1 - 'Assembly Point Location': Arrows or markers pointing to the assembly area. Tag 2 - 'Emergency Instructions': A depiction of emergency personnel providing guidance to evacuees.</data>
        </edge>
        <edge source="Decision Point 4" target="Incorrect Choice Decision Point 4">
        <data key="d0">Stay close to the building to watch what happens.</data>
        </edge>
        <edge source="Incorrect Choice Decision Point 4" target="Incorrect Choice Consequence Decision Point 4">
        <data key="d0">Emergency personnel usher you away for your safety, delaying their work.</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 4" target="Decision Point 4">
        <data key="d0">Jump back to Decision Point 4</data>
        </edge>
        <edge source="Incorrect Choice Consequence Decision Point 4" target="Media belonging to 'Incorrect Choice Consequence Decision Point 4">
        <data key="d0">Description: Video clip of emergency personnel directing people away from the building. Overlay Tags: Tag 1 - 'Safety Direction': Visual of emergency personnel guiding evacuees to safety. Tag 2 - 'Hazard Area': Highlighting the danger zone to avoid near the building.</data>
        </edge>
        <edge source="Escape Block" target="Escape Block">
        <data key="d0">Congratulations, you've navigated through the fire emergency and reached safety. Your awareness and decision-making have been crucial in this successful escape.</data>
        </edge>
    </graph>
    </graphml>
    \n\nEND OF EXAMPLE\n\n Please note that you absolutely should not give response anything else outside the graphML format since
    human will be using the generated code directly into the networkx library to run the graphML code.
    Moreover, it is absolutley mandatory and necessary for you to generate a complete graphml response such that the Graphml generated from you must close by "</graph> </graphml>" at the end of your response
    and all it's edges and nodes are also closed in the required syntax rules of graphml and all the step instructions, image tags and quiz questions be included in it since we want our graphml
    to be compilable.   
    \n\n{text}Output:""")

    GRAPHML_PROMPT_LINEAR = PromptTemplate(input_variables=['text'], template="""You are a networked intelligence helping a human track knowledge by giving providing them with
    graphml having various nodes and edges representing information about all
    relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    Extract all of the nodes and edges from the text. The nodes are connected to relevant other nodes by edges. These edges have also information in them.
    The concept is like a knowledge triplet, which is a clause that contains a subject, a predicate, and an object.
    The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.
    You are only allowed to form the node ids of words or tags enclosed in the single quotation marks ' '.
    \n\nEXAMPLE\n\n
    \nLINEAR SCENARIO:\n
    'Learning Objectives':
    - Identify the basic steps in tying shoe laces.
    - Understand the importance of starting with even laces for effective tying.
    - Learn to create and manipulate loops to form a basic shoe lace knot.
    - Recognize different styles of laces and their tying techniques through media interaction.
    - Apply knowledge through quiz questions to reinforce learning of the shoe lace tying process.                                     
    'Topic': How to Tie Shoe Laces
    [Media belonging to 'Topic', Image Description: An array of shoes with different styles of laces displayed on a white background. Overlay Tags: Tag 1 - 'Flat Laces', Tag 2 - 'Round Laces', Tag 3 - 'Colorful Laces'. Each tag, when clicked, leads to a short video showcasing the lacing technique for that style.]
    'Point 1': Ensure your shoe laces are even on both sides. Hold the ends of your laces, making sure they are of equal length.
    [Media belonging to 'Point 1', Video Description: A step-by-step guide on evening out your shoe laces before beginning the tying process. Overlay Tags: Tag 1 - 'Holding Technique', shows the correct way to hold laces. Tag 2 - 'Measuring Evenness', demonstrates how to ensure laces are even.]
    'Point 2': Cross the right lace over the left, pulling it underneath and through the loop formed, creating a simple knot.
    'Question 1 Point 2': What is the first action in tying shoe laces?
    Options: A. Make a bow. B. Cross the laces. C. Tie a double knot.
    Correct Answer: B. Cross the laces.
    'Point 3': Create a loop with each lace, holding them up to form what looks like two bunny ears.
    [Media belonging to 'Point 3', Image Description: Close-up of hands holding up two loops in the laces against a neutral background. Overlay Tags: Tag 1 - 'Loop Formation', detailed instructions on creating perfect loops. Tag 2 - 'Bunny Ear Technique', tips for maintaining loop shape.]
    'Point 4': Cross the two loops, pulling one loop through the space between them to form a secure knot.
    [Media belonging to 'Point 4', Video Description: Demonstrating the technique for crossing loops and securing the knot. Overlay Tags: Tag 1 - 'Crossing Loops', shows the crossing action. Tag 2 - 'Securing the Knot', provides additional tips for knot tightness.]
    'Quiz':
    'Question 1 Quiz Point 1': Which point involves checking if the laces are even?
    Options: A. Point 1. B. Point 2. C. Point 3.
    Correct Answer: A. Point 1.
    Score: 5 points
    'Question 2 Quiz Point 2': What shape do your hands make with the laces in Point 3?
    Options: A. Straight line. B. Bunny ears. C. Circle.
    Correct Answer: B. Bunny ears.
    Score: 5 points
    'Question 3 Quiz Point 4': How do you secure the knot?
    Options: A. By tying the loops together. B. By making another basic knot. C. No need to secure, it's already done.
    Correct Answer: A. By tying the loops together.
    Score: 5 points
    'Total Score: 15 points'
                                   
    Output:\n
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d0" for="edge" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="undirected">
        <node id="Learning Objectives"/>
        <node id="Topic"/>
        <node id="Media belonging to 'Topic'"/>
        <node id="Point 1"/>
        <node id="Media belonging to 'Point 1'"/>
        <node id="Point 2"/>
        <node id="Question 1 Point 2"/>
        <node id="Point 3"/>
        <node id="Media belonging to 'Point 3'"/>
        <node id="Point 4"/>
        <node id="Media belonging to 'Point 4'"/>
        <node id="Quiz"/>
        <node id="Question 1 Quiz Point 1"/>
        <node id="Question 2 Quiz Point 2"/>
        <node id="Question 3 Quiz Point 4"/>
        <node id="Total Score"/>
        <edge source="Learning Objectives" target="Topic">
        <data key="d0">- Identify the basic steps in tying shoe laces.
        - Understand the importance of starting with even laces for effective tying.
        - Learn to create and manipulate loops to form a basic shoe lace knot.
        - Recognize different styles of laces and their tying techniques through media interaction.
        - Apply knowledge through quiz questions to reinforce learning of the shoe lace tying process.</data>
        </edge>                                                                             
        <edge source="Topic" target="Topic">
        <data key="d0">How to Tie Shoe Laces</data>
        </edge>
        <edge source="Topic" target="Media belonging to 'Topic'">
        <data key="d0"> Image Description: An array of shoes with different styles of laces displayed on a white background. Overlay Tags: Tag 1 - 'Flat Laces', Tag 2 - 'Round Laces', Tag 3 - 'Colorful Laces'. Each tag, when clicked, leads to a short video showcasing the lacing technique for that style.</data>
        </edge>
        <edge source="Topic" target="Point 1">
        <data key="d0">Ensure your shoe laces are even on both sides. Hold the ends of your laces, making sure they are of equal length.</data>
        </edge>
        <edge source="Point 1" target="Media belonging to 'Point 1'">
        <data key="d0"> Video Description: A step-by-step guide on evening out your shoe laces before beginning the tying process. Overlay Tags: Tag 1 - 'Holding Technique', shows the correct way to hold laces. Tag 2 - 'Measuring Evenness', demonstrates how to ensure laces are even.</data>
        </edge>
        <edge source="Point 1" target="Point 2">
        <data key="d0">Cross the right lace over the left, pulling it underneath and through the loop formed, creating a simple knot.</data>
        </edge>
        <edge source="Point 2" target="Question 1 Point 2">
        <data key="d0">What is the first action in tying shoe laces? Options: A. Make a bow. B. Cross the laces. C. Tie a double knot. Correct Answer: B. Cross the laces.</data>
        </edge>
        <edge source="Point 2" target="Point 3">
        <data key="d0"> Create a loop with each lace, holding them up to form what looks like two bunny ears.</data>
        </edge>
        <edge source="Point 3" target="Media belonging to 'Point 3'">
        <data key="d0"> Image Description: Close-up of hands holding up two loops in the laces against a neutral background. Overlay Tags: Tag 1 - 'Loop Formation', detailed instructions on creating perfect loops. Tag 2 - 'Bunny Ear Technique', tips for maintaining loop shape.</data>
        </edge>
        <edge source="Point 3" target="Point 4">
        <data key="d0"> Cross the two loops, pulling one loop through the space between them to form a secure knot.</data>
        </edge>
        <edge source="Point 4" target="Media belonging to 'Point 4'">
        <data key="d0"> Video Description: Demonstrating the technique for crossing loops and securing the knot. Overlay Tags: Tag 1 - 'Crossing Loops', shows the crossing action. Tag 2 - 'Securing the Knot', provides additional tips for knot tightness.</data>
        </edge>
        <edge source="Quiz" target="Question 1 Quiz Point 1">
        <data key="d0">Which point involves checking if the laces are even?    Options: A. Point 1. B. Point 2. C. Point 3. Correct Answer: A. Point 1. Score: 5 points</data>
        </edge>
        <edge source="Quiz" target="Question 2 Quiz Point 2">
        <data key="d0">What shape do your hands make with the laces in Point 3? Options: A. Straight line. B. Bunny ears. C. Circle. Correct Answer: B. Bunny ears. Score: 5 points</data>
        </edge>
        <edge source="Quiz" target="Question 3 Quiz Point 4">
        <data key="d0"> How do you secure the knot? Options: A. By tying the loops together. B. By making another basic knot. C. No need to secure, it's already done. Correct Answer: A. By tying the loops together. Score: 5 points </data>
        </edge>
        <edge source="Quiz" target="Total Score">
        <data key="d0">15 points</data>
        </edge>
    </graph>
    </graphml>
    \n\nEND OF EXAMPLE\n\n Please note that you absolutely should not give response anything else outside the graphML format since
    human will be using the generated code directly into the networkx library to run the graphML code.
    Moreover, it is absolutley mandatory and necessary for you to generate a complete graphml response such that the Graphml generated from you must close by "</graph> </graphml>" at the end of your response
    and all it's edges and nodes are also closed in the required syntax rules of graphml and all the step instructions, image tags and quiz questions be included in it since we want our graphml
    to be compilable.   
    \n\n{text}Output:""")

    GRAPHML_PROMPT_SELF_EXPLORATORY = PromptTemplate(input_variables=['text'], template="""You are a networked intelligence helping a human track knowledge by giving providing them with
    graphml having various nodes and edges representing information about all
    relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    Extract all of the nodes and edges from the text. The nodes are connected to relevant other nodes by edges. These edges have also information in them.
    The concept is like a knowledge triplet, which is a clause that contains a subject, a predicate, and an object.
    The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.
    You are only allowed to form the node ids of words or tags enclosed in the single quotation marks ' '.
    \n\nEXAMPLE\n\n
    \nSelf-Exploratory Scenario:\n
    'Learning Objectives':
    - Differentiate between various types of renewable energy sources, such as wind and solar energy.
    - Understand the basic mechanisms behind how wind turbines and solar panels generate electricity.
    - Recognize the environmental and societal benefits of transitioning to renewable energy sources.
    - Engage with interactive media to explore the technical and environmental aspects of renewable energies.
    - Apply critical thinking to assess the impact of renewable energy on reducing carbon footprint and greenhouse gas emissions.
    'Topic': Introduction to Renewable Energy
    'Scenario':
    The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.
    [Media belonging to 'Scenario', Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.]
    'Decision Point Topic': Choose a renewable energy source to explore how it works and its benefits.
    'Topic 1 'Decision Point Topic': Wind Energy
    'Topic 1 Step 1': Understanding how wind turbines convert wind into electricity.
    [Media belonging to 'Topic 1 Step 1': Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.]
    'Topic 1 Step 2': The environmental impact and benefits of wind energy.
    [Media belonging to 'Topic 1 Step 2': Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 1 Decision Point Topic)'?
    'Quiz Topic 1 Decision Point Topic':
    'Question 1 Quiz Topic 1 Decision Point Topic': What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points
    'Question 2 Quiz Topic 1 Decision Point Topic': True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points

    'Topic 2 Decision Point Topic': Solar Energy
    'Topic 2 Step 1': How solar panels convert sunlight into electrical energy.
    [Media belonging to 'Topic 2 Step 1': Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.]
    'Topic 2 Step 2': The role of solar energy in powering homes and businesses.
    [Media belonging to 'Topic 2 Step 2': Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.]
    'Question 1 Topic 2 Step 2': What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic
    'Topic 2 Step 3': Installation and maintenance of solar panel systems.
    [Media belonging to 'Topic 2 Step 3': Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.]
    'Topic 2 Step 4': Future innovations in solar technology.
    [Media belonging to 'Topic 2 Step 4': Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.]
    'Explore More': 'YES (Jump back to Decision Point Topic)' or 'NO (Move on to Quiz Topic 2 Decision Point Topic)'?
    'Quiz Topic 2 Decision Point Topic':
    'Question 1 Quiz Topic 2 Decision Point Topic': Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points
    'Question 2 Quiz Topic 2 Decision Point Topic': True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points
    'Question 3 Quiz Topic 2 Decision Point Topic': Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points                                   
    
    Output:\n
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d0" for="edge" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="undirected">
        <!-- Nodes -->
        <node id="Learning Objectives"/>
        <node id="Topic"/>
        <node id="Scenario"/>
        <node id="Media belonging to Scenario"/>
        <node id="Decision Point Topic"/>
        <node id="Topic 1 Decision Point Topic"/>
        <node id="Topic 1 Step 1"/>
        <node id="Media belonging to Topic 1 Step 1"/>
        <node id="Topic 1 Step 2"/>
        <node id="Media belonging to Topic 1 Step 2"/>
        <node id="Quiz Topic 1 Decision Point Topic"/>
        <node id="Question 1 Quiz Topic 1 Decision Point Topic"/>
        <node id="Question 2 Quiz Topic 1 Decision Point Topic"/>
        <node id="Topic 2 Decision Point Topic"/>
        <node id="Topic 2 Step 1"/>
        <node id="Media belonging to Topic 2 Step 1"/>
        <node id="Topic 2 Step 2"/>
        <node id="Media belonging to Topic 2 Step 2"/>
        <node id="Question 1 Topic 2 Step 2"/>
        <node id="Topic 2 Step 3"/>
        <node id="Media belonging to Topic 2 Step 3"/>
        <node id="Topic 2 Step 4"/>
        <node id="Media belonging to Topic 2 Step 4"/>
        <node id="Quiz Topic 2 Decision Point Topic"/>
        <node id="Question 1 Quiz Topic 2 Decision Point Topic"/>
        <node id="Question 2 Quiz Topic 2 Decision Point Topic"/>
        <node id="Question 3 Quiz Topic 2 Decision Point Topic"/>
        <!-- Edges -->                                        
        <edge source="Learning Objectives" target="Topic">
        <data key="d0">- Differentiate between various types of renewable energy sources, such as wind and solar energy.
        - Understand the basic mechanisms behind how wind turbines and solar panels generate electricity.
        - Recognize the environmental and societal benefits of transitioning to renewable energy sources.
        - Engage with interactive media to explore the technical and environmental aspects of renewable energies.
        - Apply critical thinking to assess the impact of renewable energy on reducing carbon footprint and greenhouse gas emissions.
        </data>
        </edge>                                    
        <edge source="Topic" target="Topic">
        <data key="d0">Introduction to Renewable Energy</data>
        </edge>
        <edge source="Topic" target="Scenario">
        <data key="d0">The world is shifting towards renewable energy sources to combat climate change and reduce greenhouse gas emissions. This scenario explores different types of renewable energy, how they are harnessed, and their impact on the environment and society.</data>
        </edge>
        <edge source="Scenario" target="Media belonging to Scenario">
        <data key="d0"> Description: An aerial view of a green field with a diverse array of renewable energy sources like solar panels and wind turbines spread across the landscape. Overlay Tags: Tag 1 - 'Solar Energy Overview': Brief video on the basics of solar power generation and its significance. Tag 2 - 'Wind Power Fundamentals': Interactive animation detailing how wind turbines harness wind to produce electricity.</data>
        </edge>
        <edge source="Scenario" target="Decision Point Topic">
        <data key="d0">Choose a renewable energy source to explore how it works and its benefits.</data>
        </edge>
        <edge source="Decision Point Topic" target="Topic 1 Decision Point Topic">
        <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
        </edge>
        <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 1">
        <data key="d0">Understanding how wind turbines convert wind into electricity</data>
        </edge>
        <edge source="Topic 1 Step 1" target="Media belonging to Topic 1 Step 1">
        <data key="d0"> Description: A detailed cross-section animation of a wind turbine, showing the rotor, shaft, and generator. Overlay Tags: Tag 1 - 'Turbine Mechanics': Animated breakdown of the turbine's components and their functions. Tag 2 - 'Energy Conversion': Explainer video on the process of converting wind into electrical energy.</data>
        </edge>
        <edge source="Topic 1 Decision Point Topic" target="Topic 1 Step 2">
        <data key="d0">The environmental impact and benefits of wind energy</data>
        </edge>
        <edge source="Topic 1 Step 2" target="Media belonging to Topic 1 Step 2">
        <data key="d0"> Description: An infographic that contrasts the CO2 emissions from wind energy with those from fossil fuels. Overlay Tags: Tag 1 - 'Emission Reduction': Graphical data on how wind energy reduces overall carbon footprint. Tag 2 - 'Renewable Benefits': A quick guide on the positive environmental impacts of adopting wind energy.</data>
        </edge>
        <edge source="Topic 1 Step 2" target="Quiz Topic 1 Decision Point Topic">
        <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 1 Decision Point Topic)?</data>
        </edge>
        <edge source="Quiz Topic 1 Decision Point Topic" target="Question 1 Quiz Topic 1 Decision Point Topic">
        <data key="d0">What part of the wind turbine captures wind energy? (Blades/Rotor) Correct Answer: Blades Score: 10 points</data>
        </edge>
        <edge source="Quiz Topic 1 Decision Point Topic" target="Question 2 Quiz Topic 1 Decision Point Topic">
        <data key="d0">True or False: Wind energy produces greenhouse gases during electricity generation. Correct Answer: False Score: 10 points</data>
        </edge>
        <edge source="Decision Point Topic" target="Topic 2 Decision Point Topic">
        <data key="d0">Choose a renewable energy source to explore how it works and its benefits</data>
        </edge>
        <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 1">
        <data key="d0">How solar panels convert sunlight into electrical energy</data>
        </edge>
        <edge source="Topic 2 Step 1" target="Media belonging to Topic 2 Step 1">
        <data key="d0"> Description: A video explaining the photovoltaic effect and the operation of solar cells within a panel. Overlay Tags: Tag 1 - 'Photovoltaic Effect': Video tutorial on how sunlight is converted into electricity by solar panels. Tag 2 - 'Solar Cell Function': Interactive diagram of a solar cell with details on its components and how they work together.</data>
        </edge>
        <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 2">
        <data key="d0">The role of solar energy in powering homes and businesses</data>
        </edge>
        <edge source="Topic 2 Step 2" target="Media belonging to Topic 2 Step 2">
        <data key="d0"> Description: A case study presentation of a solar-powered smart home, emphasizing energy savings and efficiency. Overlay Tags: Tag 1 - 'Smart Home Energy': Virtual tour of a home powered by solar energy, highlighting key features and benefits. Tag 2 - 'Cost Savings': Infographic on the economic advantages of solar energy for households and businesses.</data>
        </edge>
        <edge source="Topic 2 Step 2" target="Question 1 Topic 2 Step 2">
        <data key="d0">What is the name of the effect by which solar panels generate electricity? Correct Answer: Photovoltaic</data>
        </edge>
        <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 3">
        <data key="d0">Installation and maintenance of solar panel systems</data>
        </edge>
        <edge source="Topic 2 Step 3" target="Media belonging to Topic 2 Step 3">
        <data key="d0"> Description: A detailed visual guide showcasing the step-by-step process of installing rooftop solar panels, including the tools required, safety measures, and best practices for optimal installation. Overlay Tags: Tag 1 - 'Installation Process': A comprehensive video tutorial guiding through the safe and efficient installation of rooftop solar panels. Tag 2 - 'Maintenance Tips': A series of tips and best practices for maintaining solar panels to ensure their longevity and maximum efficiency.</data>
        </edge>
        <edge source="Topic 2 Decision Point Topic" target="Topic 2 Step 4">
        <data key="d0">Future innovations in solar technology</data>
        </edge>
        <edge source="Topic 2 Step 4" target="Media belonging to Topic 2 Step 4">
        <data key="d0"> Description: Concept art and visualizations of next-generation solar technologies, highlighting transparent solar panels that can be integrated into windows and flexible solar panels that can be applied to various surfaces for more versatile use. Overlay Tags: Tag 1 - 'Transparent Solar Panels': An interactive exploration of the technology behind transparent solar panels, their potential applications, and how they can transform urban and residential environments. Tag 2 - 'Flexible Solar Technology': A deep dive into the development and benefits of flexible solar panels, showcasing their potential for integration into everyday objects and their role in expanding the accessibility of solar power.</data>
        </edge>
        <edge source="Topic 2 Step 4" target="Quiz Topic 2 Decision Point Topic">
        <data key="d0">Explore More: YES (Jump back to Decision Point Topic) or NO (Move on to Quiz Topic 2 Decision Point Topic)?</data>
        </edge>
        <edge source="Quiz Topic 2 Decision Point Topic" target="Question 1 Quiz Topic 2 Decision Point Topic">
        <data key="d0">Solar panels are most efficient in which type of climate? Correct Answer: Sunny and cool Score: 10 points</data>
        </edge>
        <edge source="Quiz Topic 2 Decision Point Topic" target="Question 2 Quiz Topic 2 Decision Point Topic">
        <data key="d0">True or False: Solar panels cannot produce electricity on cloudy days. Correct Answer: False Score: 10 points</data>
        </edge>
        <edge source="Quiz Topic 2 Decision Point Topic" target="Question 3 Quiz Topic 2 Decision Point Topic">
        <data key="d0">Fill in the blank: The ______ effect is crucial for solar panels to convert sunlight into electricity. Correct Answer: Photovoltaic Score: 10 points</data>
        </edge>
    </graph>
    </graphml>
    \n\nEND OF EXAMPLE\n\n Please note that you absolutely should not give response anything else outside the graphML format since
    human will be using the generated code directly into the networkx library to run the graphML code.
    Moreover, it is absolutley mandatory and necessary for you to generate a complete graphml response such that the Graphml generated from you must close by "</graph> </graphml>" at the end of your response
    and all it's edges and nodes are also closed in the required syntax rules of graphml and all the step instructions, image tags and quiz questions be included in it since we want our graphml
    to be compilable.   
    \n\n{text}Output:""")

    GRAPHML_PROMPT_SIMULATION = PromptTemplate(input_variables=['text'], template="""You are a networked intelligence helping a human track knowledge by giving providing them with
    graphml having various nodes and edges representing information about all
    relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    Extract all of the nodes and edges from the text. The nodes are connected to relevant other nodes by edges. These edges have also information in them.
    The concept is like a knowledge triplet, which is a clause that contains a subject, a predicate, and an object.
    The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.
    You are only allowed to form the node ids of words or tags enclosed in the single quotation marks ' '.
    \n\nEXAMPLE\n\n
    \nSIMULATION SCENARIO:\n
    'Learning Objectives':
    - Navigate through a corporate archive section to understand its layout and the function of different sub-sections.
    - Identify the operational processes within the printing office, including the use and benefits of specific equipment like scanners and printers.
    - Explore the management and correspondence functions within the head office, including team organization and documentation.
    - Investigate the library section to learn how publications are stored, accessed, and managed digitally and physically.
    - Apply decision-making skills to explore detailed aspects of archive management, such as digital storage solutions and publication cataloging.
    'Topic': Introduction to the Archive Section
    'Scenario': You are in the main lounge of a company building on your first day for an office tour. You are being shown an Archive Section where there different sub-sections situated at the left, right and upstairs from the main lounge.
    [Media belonging to 'Scenario', Description: A 360-degree image of a corporate lounge with paths leading to the left, right, and upstairs. Overlay Tags: Tag 1 - 'Left Path': An arrow pointing left with a caption: "To the Printing Office". Tag 2 - 'Right Path': An arrow pointing right with a caption: "To the Library". Tag 3 - 'Upstairs': An arrow pointing upwards with a caption: "To the Head Office".]
    'Decision Point Main': There are 3 paths to choose from the main lounge. Left, Right and Upstairs.
    'Timer Decision Point Main': 1 minutes
    'Choice 1 Decision Point Main': Left path.
    'Choice 1 Consequence Decision Point Main': You reach the printing office of Archive Section when you move left from the lounge. 'Move on to Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point Main', Description: Picture of an office with various printers and a large dedicated scanner. Overlay Tags: Tag 1 - 'Printer': Close-up on a printer with a caption: "60 Copies per Minute". Tag 2 - 'Dedicated Scanner': Close-up on a scanner with a caption: "Scans up to A0 size paper".]
    'Choice 2 Decision Point Main': Upstairs.
    'Choice 2 Consequence Decision Point Main': You reach to the head office of Archive Section where management and correspondence is carried out. 'Move on to Decision Point 2'.
    [Media belonging to 'Choice 2 Consequence Decision Point Main', Description: A picture showing an office that may belong to a manager, with documents and a computer on a desk. Overlay Tags: Tag 1 - 'Manager's Desk': A close-up on the desk with a caption: "Team State Document".]
    'Choice 3 Decision Point Main': Right path.
    'Choice 3 Consequence Decision Point Main': You reach the library of the Archive Section. 'Move on to Decision Point 3'.
    [Media belonging to 'Choice 3 Consequence Decision Point Main', Description: Picture of a library with shelves full of books and a computer setup for accessing records. Overlay Tags: Tag 1 - 'Computer Records': Zoom in on the computer screen with a caption: "Access Publication Records Here". Tag 2 - 'Soft Storage Cabinets': Focus on closed cabinets with a caption: "Contains Digital Archives".]

    'Decision Point 1': The printing office is used to print the publications. They are given to the library subsection after printing. There are 2 machines in the printing office, a large dedicated scanner and a large printer.
    'Timer Decision Point 1': 5 minutes
    'Choice 1 Decision Point 1': Dedicated Scanner.
    'Choice 1 Consequence Decision Point 1': The dedicated scanner can scan upto A0 size paper, mostly used for scanning large engineering blue-prints. 'Move on to Result Choice 1 Decision Point 1'.
    [Media belonging to 'Choice 1 Consequence Decision Point 1', Description: Video of a large drawing being scanned by a scanner. Overlay Tags: Tag 1 - 'Scanning Process': Detailed video showing how the dedicated scanner operates, including paper feeding, scanning mechanism, and output. Tag 2 - 'Blueprint Scanning': Tips on preparing and positioning large engineering blueprints for scanning.]
    'Result Choice 1 Decision Point 1': Score 5
    'Choice 2 Decision Point 1': Printer. 
    'Choice 2 Consequence Decision Point 1': This printer can print 60 Copies per Minute. 'Move on to Result Choice 2 Decision Point 1'.
    [Media belonging to 'Choice 2 Consequence Decision Point 1', Description: Video of a large printer in operation, showing rapid printing. Overlay Tags: Tag 1 - 'Printer Capabilities': Explanation of printer features, including speed, paper handling, and quality settings. Tag 2 - 'High-Volume Printing': Best practices for managing large print jobs efficiently.]
    'Result Choice 2 Decision Point 1': Score 5

    'Decision Point 2': The head office is used for correspondence, and management.
    'Timer Decision Point 2': 5 minutes
    'Choice 1 Decision Point 2': See the team state document.
    'Choice 1 Consequence Decision Point 2': The team state document shows all the employees and their data with work responsibilities and leave records. 'Move on to Result Choice 1 Decision Point 2'.
    [Media belonging to 'Choice 1 Consequence Decision Point 2', Description: Picture of an office desk with various documents and a computer screen displaying an employee database. Overlay Tags: Tag 1 - 'Employee Database': Detailed view of the database interface with functionalities highlighted. Tag 2 - 'Document Overview': Close-up on key documents that manage team responsibilities and leave schedules.]
    'Result Choice 1 Decision Point 2': Score 10

    'Decision Point 3': The library is used to keep all the record of publications printed in the printing section.
    'Timer Decision Point 3': 10 minutes
    'Choice 1 Decision Point 3': View computer records of publications.
    'Choice 1 Consequence Decision Point 3': The computer records contains record of available publications and their locations on the shelves. 'Move on to Result Choice 1 Decision Point 3'.
    [Media belonging to 'Choice 1 Consequence Decision Point 3', Description: An image of a computer screen displaying a digital catalog of publications with search functionality and shelf location information. Overlay Tags: Tag 1 - 'Digital Catalog': Interactive demo on navigating the digital records to find publications. Tag 2 - 'Locating Publications': Instructions on how to use the catalog to find the exact shelf location of a book or document.]
    'Result Choice 1 Decision Point 3': Score 5
    'Choice 2 Decision Point 3': View the closed soft storage cabinets.
    'Choice 2 Consequence Decision Point 3': The cabinet is opened for you to see inside. 'Move on to Decision Point 4'.
    [Media belonging to 'Choice 2 Consequence Decision Point 3', Description: Picture of an opened storage cabinet revealing various labeled compartments and digital storage media. Overlay Tags: Tag 1 - 'Storage Organization': Overview of how the cabinet is organized for efficient storage and retrieval. Tag 2 - 'Digital Media Storage': Insights into the preservation and categorization of digital archives.]

    'Decision Point 4': The cabinets contains various disks that stores the content of the publications in soft form, to be given with the publications at the time of issue to customers.
    'Timer Decision Point 4': 3 minutes
    'Choice 1 Decision Point 4': Inspect a disk in the cabinet.
    'Choice 1 Consequence Decision Point 4': On the disk, information is given about what publication this disk belongs to and where the location of that publication in the shelves is. 'Move on to Result Choice 1 Decision Point 4'.
    [Media belonging to 'Choice 1 Consequence Decision Point 4', Description: A close-up of a disk with labels indicating the publication it accompanies and a map showing its shelf location in the library. Overlay Tags: Tag 1 - 'Disk Content Overview': A quick guide on the information provided on the disk, including the publication it belongs to. Tag 2 - 'Finding Publications': Instructions on using the disk's label to locate the physical publication in the library.]
    'Result Choice 1 Decision Point 4': Score 20
    
    Output:\n
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d0" for="edge" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="undirected">
        <!-- Nodes -->
        <node id="Learning Objectives"/>
        <node id="Topic"/>
        <node id="Scenario"/>
        <node id="Media belonging to Scenario"/>
        <node id="Decision Point Main"/>
        <node id="Timer Decision Point Main"/>
        <node id="Choice 1 Decision Point Main"/>
        <node id="Choice 1 Consequence Decision Point Main"/>
        <node id="Media belonging to Choice 1 Consequence Decision Point Main"/>
        <node id="Choice 2 Decision Point Main"/>
        <node id="Choice 2 Consequence Decision Point Main"/>
        <node id="Media belonging to Choice 2 Consequence Decision Point Main"/>
        <node id="Choice 3 Decision Point Main"/>
        <node id="Choice 3 Consequence Decision Point Main"/>
        <node id="Media belonging to Choice 3 Consequence Decision Point Main"/>
        <node id="Decision Point 1"/>
        <node id="Timer Decision Point 1"/>
        <node id="Choice 1 Decision Point 1"/>
        <node id="Choice 1 Consequence Decision Point 1"/>
        <node id="Media belonging to Choice 1 Consequence Decision Point 1"/>
        <node id="Result Choice 1 Decision Point 1"/>
        <node id="Choice 2 Decision Point 1"/>
        <node id="Choice 2 Consequence Decision Point 1"/>
        <node id="Media belonging to Choice 2 Consequence Decision Point 1"/>
        <node id="Result Choice 2 Decision Point 1"/>
        <node id="Decision Point 2"/>
        <node id="Timer Decision Point 2"/>
        <node id="Choice 1 Decision Point 2"/>
        <node id="Choice 1 Consequence Decision Point 2"/>
        <node id="Media belonging to Choice 1 Consequence Decision Point 2"/>
        <node id="Result Choice 1 Decision Point 2"/>
        <node id="Decision Point 3"/>
        <node id="Timer Decision Point 3"/>
        <node id="Choice 1 Decision Point 3"/>
        <node id="Choice 1 Consequence Decision Point 3"/>
        <node id="Media belonging to Choice 1 Consequence Decision Point 3"/>
        <node id="Result Choice 1 Decision Point 3"/>
        <node id="Choice 2 Decision Point 3"/>
        <node id="Choice 2 Consequence Decision Point 3"/>
        <node id="Media belonging to Choice 2 Consequence Decision Point 3"/>
        <node id="Decision Point 4"/>
        <node id="Timer Decision Point 4"/>
        <node id="Choice 1 Decision Point 4"/>
        <node id="Choice 1 Consequence Decision Point 4"/>
        <node id="Media belonging to Choice 1 Consequence Decision Point 4"/>
        <node id="Result Choice 1 Decision Point 4"/>
        <!-- Edges -->
        <edge source="Learning Objectives" target="Topic">
        <data key="d0">- Navigate through a corporate archive section to understand its layout and the function of different sub-sections.
        - Identify the operational processes within the printing office, including the use and benefits of specific equipment like scanners and printers.
        - Explore the management and correspondence functions within the head office, including team organization and documentation.
        - Investigate the library section to learn how publications are stored, accessed, and managed digitally and physically.
        - Apply decision-making skills to explore detailed aspects of archive management, such as digital storage solutions and publication cataloging.
        </data>
        </edge>
        <edge source="Topic" target="Topic">
        <data key="d0">Introduction to the Archive Section</data>
        </edge>
        <edge source="Topic" target="Scenario">
        <data key="d0">You are in the main lounge of a company building on your first day for an office tour. You are being shown an Archive Section where there different sub-sections situated at the left, right and upstairs from the main lounge.</data>
        </edge>
        <edge source="Topic" target="Media belonging to Scenario">
        <data key="d0">Description: A 360-degree image of a corporate lounge with paths leading to the left, right, and upstairs. Overlay Tags: Tag 1 - 'Left Path': An arrow pointing left with a caption: "To the Printing Office". Tag 2 - 'Right Path': An arrow pointing right with a caption: "To the Library". Tag 3 - 'Upstairs': An arrow pointing upwards with a caption: "To the Head Office".</data>
        </edge>
        <edge source="Scenario" target="Decision Point Main">
        <data key="d0">There are 3 paths to choose from the main lounge. Left, Right and Upstairs.</data>
        </edge>
        <edge source="Decision Point Main" target="Timer Decision Point Main">
        <data key="d0">1 minutes</data>
        </edge>
        <edge source="Decision Point Main" target="Choice 1 Decision Point Main">
        <data key="d0">Left path</data>
        </edge>
        <edge source="Choice 1 Decision Point Main" target="Choice 1 Consequence Decision Point Main">
        <data key="d0">You reach the printing office of Archive Section when you move left from the lounge</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point Main" target="Decision Point 1">
        <data key="d0">Move on to Decision Point 1</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point Main" target="Media belonging to Choice 1 Consequence Decision Point Main">
        <data key="d0">Description: Picture of an office with various printers and a large dedicated scanner. Overlay Tags: Tag 1 - 'Printer': Close-up on a printer with a caption: "60 Copies per Minute". Tag 2 - 'Dedicated Scanner': Close-up on a scanner with a caption: "Scans up to A0 size paper".</data>
        </edge>
        <edge source="Decision Point Main" target="Choice 2 Decision Point Main">
        <data key="d0">Upstairs</data>
        </edge>
        <edge source="Choice 2 Decision Point Main" target="Choice 2 Consequence Decision Point Main">
        <data key="d0">You reach to the head office of Archive Section where management and correspondence is carried out</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point Main" target="Decision Point 2">
        <data key="d0">Move on to Decision Point 2</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point Main" target="Media belonging to Choice 2 Consequence Decision Point Main">
        <data key="d0">Description: A picture showing an office that may belong to a manager, with documents and a computer on a desk. Overlay Tags: Tag 1 - 'Manager's Desk': A close-up on the desk with a caption: "Team State Document".</data>
        </edge>
        <edge source="Decision Point Main" target="Choice 3 Decision Point Main">
        <data key="d0">Right path</data>
        </edge>
        <edge source="Choice 3 Decision Point Main" target="Choice 3 Consequence Decision Point Main">
        <data key="d0">You reach the library of the Archive Section</data>
        </edge>
        <edge source="Choice 3 Consequence Decision Point Main" target="Decision Point 3">
        <data key="d0">Move on to Decision Point 3</data>
        </edge>
        <edge source="Choice 3 Consequence Decision Point Main" target="Media belonging to Choice 3 Consequence Decision Point Main">
        <data key="d0">Description: Picture of a library with shelves full of books and a computer setup for accessing records. Overlay Tags: Tag 1 - 'Computer Records': Zoom in on the computer screen with a caption: "Access Publication Records Here". Tag 2 - 'Soft Storage Cabinets': Focus on closed cabinets with a caption: "Contains Digital Archives".</data>
        </edge>

        <edge source="Decision Point 1" target="Timer Decision Point 1">
        <data key="d0">5 minutes</data>
        </edge>
        <edge source="Decision Point 1" target="Choice 1 Decision Point 1">
        <data key="d0">Dedicated Scanner</data>
        </edge>
        <edge source="Choice 1 Decision Point 1" target="Choice 1 Consequence Decision Point 1">
        <data key="d0">The dedicated scanner can scan up to A0 size paper, mostly used for scanning large engineering blueprints</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 1" target="Media belonging to Choice 1 Consequence Decision Point 1">
        <data key="d0">Description: Video of a large drawing being scanned by a scanner. Overlay Tags: Tag 1 - 'Scanning Process': Detailed video showing how the dedicated scanner operates, including paper feeding, scanning mechanism, and output. Tag 2 - 'Blueprint Scanning': Tips on preparing and positioning large engineering blueprints for scanning.</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 1" target="Result Choice 1 Decision Point 1">
        <data key="d0">Score 5</data>
        </edge>
        <edge source="Decision Point 1" target="Choice 2 Decision Point 1">
        <data key="d0">Printer</data>
        </edge>
        <edge source="Choice 2 Decision Point 1" target="Choice 2 Consequence Decision Point 1">
        <data key="d0">This printer can print 60 Copies per Minute</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point 1" target="Media belonging to Choice 2 Consequence Decision Point 1">
        <data key="d0">Description: Video of a large printer in operation, showing rapid printing. Overlay Tags: Tag 1 - 'Printer Capabilities': Explanation of printer features, including speed, paper handling, and quality settings. Tag 2 - 'High-Volume Printing': Best practices for managing large print jobs efficiently.</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point 1" target="Result Choice 2 Decision Point 1">
        <data key="d0">Score 5</data>
        </edge>

        <edge source="Decision Point 2" target="Timer Decision Point 2">
        <data key="d0">5 minutes</data>
        </edge>
        <edge source="Decision Point 2" target="Choice 1 Decision Point 2">
        <data key="d0">See the team state document</data>
        </edge>
        <edge source="Choice 1 Decision Point 2" target="Choice 1 Consequence Decision Point 2">
        <data key="d0">The team state document shows all the employees and their data with work responsibilities and leave records</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 2" target="Media belonging to Choice 1 Consequence Decision Point 2">
        <data key="d0">Description: Picture of an office desk with various documents and a computer screen displaying an employee database. Overlay Tags: Tag 1 - 'Employee Database': Detailed view of the database interface with functionalities highlighted. Tag 2 - 'Document Overview': Close-up on key documents that manage team responsibilities and leave schedules.</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 2" target="Result Choice 1 Decision Point 2">
        <data key="d0">Score 10</data>
        </edge>

        <edge source="Decision Point 3" target="Timer Decision Point 3">
        <data key="d0">10 minutes</data>
        </edge>
        <edge source="Decision Point 3" target="Choice 1 Decision Point 3">
        <data key="d0">View computer records of publications</data>
        </edge>
        <edge source="Choice 1 Decision Point 3" target="Choice 1 Consequence Decision Point 3">
        <data key="d0">Description: An image of a computer screen displaying a digital catalog of publications with search functionality and shelf location information. Overlay Tags: Tag 1 - 'Digital Catalog': Interactive demo on navigating the digital records to find publications. Tag 2 - 'Locating Publications': Instructions on how to use the catalog to find the exact shelf location of a book or document.</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 3" target="Media belonging to Choice 1 Consequence Decision Point 3">
        <data key="d0">Description: An image of a computer screen displaying a digital catalog of publications with search functionality and shelf location information. Overlay Tags: Tag 1 - 'Digital Catalog': Interactive demo on navigating the digital records to find publications. Tag 2 - 'Locating Publications': Instructions on how to use the catalog to find the exact shelf location of a book or document.</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 3" target="Result Choice 1 Decision Point 3">
        <data key="d0">Score 5</data>
        </edge>
        <edge source="Decision Point 3" target="Choice 2 Decision Point 3">
        <data key="d0">View the closed soft storage cabinets</data>
        </edge>
        <edge source="Choice 2 Decision Point 3" target="Choice 2 Consequence Decision Point 3">
        <data key="d0">The cabinet is opened for you to see inside</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point 3" target="Decision Point 4">
        <data key="d0">Move on to Decision Point 4</data>
        </edge>
        <edge source="Choice 2 Consequence Decision Point 3" target="Media belonging to Choice 2 Consequence Decision Point 3">
        <data key="d0">Description: Picture of an opened storage cabinet revealing various labeled compartments and digital storage media. Overlay Tags: Tag 1 - 'Storage Organization': Overview of how the cabinet is organized for efficient storage and retrieval. Tag 2 - 'Digital Media Storage': Insights into the preservation and categorization of digital archives.</data>
        </edge>
        <edge source="Decision Point 4" target="Timer Decision Point 4">
        <data key="d0">3 minutes</data>
        </edge>
        <edge source="Decision Point 4" target="Choice 1 Decision Point 4">
        <data key="d0">Inspect a disk in the cabinet</data>
        </edge>
        <edge source="Choice 1 Decision Point 4" target="Choice 1 Consequence Decision Point 4">
        <data key="d0">On the disk, information is given about what publication this disk belongs to and where the location of that publication in the shelves is</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 4" target="Media belonging to Choice 1 Consequence Decision Point 4">
        <data key="d0">Description: A close-up of a disk with labels indicating the publication it accompanies and a map showing its shelf location in the library. Overlay Tags: Tag 1 - 'Disk Content Overview': A quick guide on the information provided on the disk, including the publication it belongs to. Tag 2 - 'Finding Publications': Instructions on using the disk's label to locate the physical publication in the library.</data>
        </edge>
        <edge source="Choice 1 Consequence Decision Point 4" target="Result Choice 1 Decision Point 4">
        <data key="d0">Score 20</data>
        </edge>
    </graph>
    </graphml>
    \n\nEND OF EXAMPLE\n\n Please note that you absolutely should not give response anything else outside the graphML format since
    human will be using the generated code directly into the networkx library to run the graphML code.
    Moreover, it is absolutley mandatory and necessary for you to generate a complete graphml response such that the Graphml generated from you must close by "</graph> </graphml>" at the end of your response
    and all it's edges and nodes are also closed in the required syntax rules of graphml and all the step instructions, image tags and quiz questions be included in it since we want our graphml
    to be compilable.   
    \n\n{text}Output:""")

    ### SEMANTIC ROUTES LOGIC ###
    linear = Route(
    name="linear",
    utterances=[
        f"linear scenario is mentioned in following= {bot_last_reply}",
    ],
    )
    escaperoom = Route(
        name="escape room",
        utterances=[
            f"escape room scenario is mentioned in following= {bot_last_reply}",
        ],
    )
    simulation = Route(
        name="simulation",
        utterances=[
            f"simulation scenario is mentioned in following= {bot_last_reply}",
        ],
    )
    selfexploratory = Route(
        name="self exploratory",
        utterances=[
            f"self exploratory scenario is mentioned in following= {bot_last_reply}",
        ],
    )
    routes = [linear, escaperoom, simulation, selfexploratory]
    encoder = OpenAIEncoder()
    rl = RouteLayer(encoder=encoder, routes=routes)
    x = rl(bot_last_reply)
    print("GraphML of NAME",x.name)
    ############################

    # llmsx = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    if x.name == 'escaperoom':
        graphml_chain = LLMChain(llm= llmsx, prompt=GRAPHML_PROMPT_ESCAPE_ROOM)
    elif x.name == 'linear':
        graphml_chain = LLMChain(llm= llmsx, prompt=GRAPHML_PROMPT_LINEAR)
    elif x.name == 'simulation':
        graphml_chain = LLMChain(llm= llmsx, prompt=GRAPHML_PROMPT_SIMULATION)
    else:
        graphml_chain = LLMChain(llm= llmsx, prompt=GRAPHML_PROMPT_SELF_EXPLORATORY)
    
    # output_graphml = graphml_chain.predict(text=bot_last_reply)

    return graphml_chain

def DRAW_GRAPH(output_graphml_generated_again, width,height):
    import networkx as nx
    import matplotlib.pyplot as plt
    import base64
    import io
    matplotlib.use('Agg')
    print(output_graphml_generated_again, width,height,"LCD SIDE")
    G = nx.read_graphml(output_graphml_generated_again)
    print("wow")
    plt.figure(figsize=(int(width),int(height)))
    options = {
        "font_size": 6,
        "linewidths": 3,
    }
    nx.draw_networkx(G, **options)
    ax = plt.gca()
    ax.margins(0.2)
    plt.axis('off')
    plt.plot()

    # Saving the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    
    # Converting the plot image to base64 for embedding in HTML
    plot_image = base64.b64encode(buffer.getvalue())
    plot_image_uri = f"data:image/png;base64,{plot_image.decode()}"
    
    return plot_image_uri

