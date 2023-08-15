"""Python file to serve as the frontend"""
import os
import pickle

import faiss
import streamlit as st
from langchain import OpenAI, PromptTemplate
from langchain.chains import VectorDBQAWithSourcesChain
from streamlit_chat import message

os.environ["OPENAI_API_KEY"] = "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"
# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

# System prompt requiring Question and Year to be extracted from the user
combine_prompt1 = """
You are a professional Bitmart customer service and a knowledgeable assistant.
Please translate the answers based on the language in question.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Think about this step by step:
- The user will ask a Question
- Reply the user's question.
- ALWAYS return a ("SOURCES") part in your answer.

Example:
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

COMBINE_PROMPT1 = PromptTemplate(
    template=combine_prompt1, input_variables=["summaries", "question"]
)

chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store,
                                            combine_prompt=COMBINE_PROMPT1)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Bitmart QA Bot", page_icon=":robot:")
st.header("Bitmart QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
