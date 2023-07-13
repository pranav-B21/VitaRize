# Import the required libraries
import openai
# import os
import streamlit as st
from streamlit_option_menu import option_menu
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Set the GPT-3 API key
openai.api_key = st.secrets["pass"]

selected_page = option_menu(
    menu_title=None,
    options=["Summarizer", "feature_2"],
    icons=["info", "info"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


def generate_summary(text, user_prompt):
    if st.button("Generate Summary", type='primary'):
        # Use GPT-3 to generate a summary of the article
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="This is the user prompt for how the text should be summarized" +
            user_prompt + "here is the text:" + text,
            temperature=0.5,
        )
        # Print the generated summary
        res = response["choices"][0]["text"]
        st.success(res)
        st.download_button('Download result', res)


def extract_text_from_pdf(file):
    print("hi")


if selected_page == "Summarizer":
    st.title("PDF Summarization")
    st.write("Upload a PDF document and enter a prompt. Then click 'Generate Summary' to generate a personalized summary.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Custom Prompt
    prompt = st.text_input("Enter a prompt")

    if uploaded_file is not None:
        # Convert PDF to text
        text = extract_text_from_pdf(uploaded_file)

        # Display original text
        st.header("Original Text")
        st.write(text)

        # Generate summary
        if prompt:
            summary = generate_summary(text, prompt)
            st.header("Summary")
            st.write(summary)
