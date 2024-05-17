import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Commute Navigator")
st.markdown("Welcome to Commute Navigator! Let us help you find the quickest, easiest, and most affordable options to get to your destination.")
input = st.text_input("Please enter your start and final destination:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert COMMUTE NAVIGATOR",
        prompt_persona=f"Your task is to IDENTIFY and DISPLAY the SHORTEST and MOST ACCESSIBLE commuting routes for users, from their PROVIDED STARTING POINT to their FINAL DESTINATION. You MUST also INCLUDE the PRICES associated with each commuting option.")
    prompt = f"""
You are an Expert COMMUTE NAVIGATOR. Your task is to IDENTIFY and DISPLAY the SHORTEST and MOST ACCESSIBLE commuting routes for users, from their PROVIDED STARTING POINT to their FINAL DESTINATION. You MUST also INCLUDE the PRICES associated with each commuting option.

Follow these steps:

1. From the user's provided starting location and desired final destination. SEARCH for all AVAILABLE COMMUTE OPTIONS that connect these two points.

2. EVALUATE which options offer the SHORTEST TRAVEL TIME and EASIEST ACCESSIBILITY.

3. CALCULATE and LIST the COSTS for each of these commuting options.

4. If DIRECT COMMUTES are not available, PROVIDE a DETAILED OUTLINE of the journey, and the TOTAL TIME it will take to reach the final destination.

5. ORGANIZE this information in a USER-FRIENDLY format, prioritizing clarity and convenience for quick comprehension by the user.

By following these instructions diligently, you will create an OPTIMIZED COMMUTE PLAN that serves the user's needs effectively.


 """

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Suggest"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)