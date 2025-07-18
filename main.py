import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import asyncio

# Page settings
st.set_page_config(page_title="Translator Agent", page_icon=":globe_with_meridians:")

st.markdown("""
<style>
    
    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #145a8a;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Please set the GEMINI_API_KEY environment variable.")
    st.stop()

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Set model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Create agent
translator_agent = Agent(
    name="Translator Agent",
    instructions="You are a translator agent. Translate the input text to the target language specified in the input.",
    model=model,
)


st.title("Translator Agent")
st.write("This is a simple translator agent that uses the Gemini API to translate text.")

user_input = st.text_input("Enter text to translate:")
target_language = st.selectbox("Select target language:", ["Spanish", "French", "German", "Chinese", "Urdu", "English","Arabic","Japanese","Korean","Russian","Italian,","Roman urdu"])

# Async translation function
async def run_translation():
    prompt = f"Translate the following text to {target_language}:\n\n{user_input}"
    return await Runner.run(
        translator_agent,
        prompt,
        run_config=config
    )
if st.button("Translate"):
    if user_input:
        with st.spinner("Translating..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_translation())
                st.text_area("Translation:", value=result.final_output, height=150)
                st.success("Translation completed!")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                loop.close()
    else:
        st.warning("Please enter some text first.")


