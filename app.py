import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

wikipidea_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(api_wrapper=wikipidea_api_wrapper)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="DuckDuckGo Search", description="Search the web using DuckDuckGo.")

st.set_page_config(page_title="Querylytic", page_icon=":robot_face:", layout="centered")
st.title("Querylytic :robot_face:")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am Querylytic, your AI assistant. Who can search the web for you and answer your questions. How can I help you today?"
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input(placeholder="Ask me anything!!!"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wikipedia]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        reponse = search_agent.run(prompt, callbacks=[st_callback])
        st.session_state.messages.append({
            "role": "assistant",
            "content": reponse
        })
    st.chat_message("assistant").write(reponse)