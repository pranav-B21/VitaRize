import openai
import streamlit as st
from streamlit_option_menu import option_menu
import PyPDF2
from transformers import pipeline
import pandas as pd
import altair as alt
import pytesseract
from PIL import Image


# pip install openai
# pip install streamlit
# pip install PyPDF2
# pip install transformers
# pip install torch
# brew install poppler-qt5
# pip install pytesseract / brew install tesseract (i would do both to be safe)
# once you get a new api key, go to secrets.toml and add it instead of pass

# run the code --> streamlit run main.py

# Set the GPT-3 API key
openai.api_key = st.secrets["pass"]

sentiment_score = {}

logo = "Design.png"
st.image(logo, use_column_width=True)

selected_page = option_menu(
    menu_title=None,
    options=["Home", "Summarizer", "Sentiment"],
    icons=["house", "info", "heart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

def generate_summary(text, temperature, length, user_prompt):
    # Use GPT-3 to generate a summary of the article
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="What do you want to learn from this paper? " +
        user_prompt + " Here is the text:" + text,
        temperature=temperature,
        max_tokens=length,
    )
    # Print the generated summary
    res = response["choices"][0]["text"]
    st.success(res)
    st.download_button('Download result', res)


def extract_text_from_img(file):
    # Convert the image into a string
    text = str(((pytesseract.image_to_string(Image.open(file)))))
    text = text.replace("-\n", "")
    return text

def extract_text_from_pdf(file):
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(file)

    # Extract text from each page
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Close the PDF file
    file.close()

    return text

def predict_sentiment(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(user_input)[0]
    return result


if selected_page == "Summarizer":
    st.title("Document Analyzer")
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    uploaded_image = st.file_uploader("Upload a PNG file", type=['png'])
    if uploaded_file is not None:
        # Convert PDF to text
        text = extract_text_from_pdf(uploaded_file)
        # Prompt settings
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        length = st.slider("Length of Summary", min_value=50,
                           max_value=500, step=50, value=100)
        user_prompt = st.text_input(
            "Ask questions about this file")

        if st.button("Generate Summary"):
            summary = generate_summary(text, temperature, length, user_prompt)
            st.header("Summary")
            st.write(summary)
    if uploaded_image is not None:
        # Convert PNG to text
        text = extract_text_from_img(uploaded_image)
        # Prompt settings
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        length = st.slider("Length of Summary", min_value=50,
                           max_value=500, step=50, value=100)
        user_prompt = st.text_input(
            "Ask questions about this file")

        if st.button("Generate Summary"):
            summary = generate_summary(text, temperature, length, user_prompt)
            st.header("Summary")
            st.write(summary)

if selected_page == "Sentiment":
    print("sentiment")
    st.title("Sentiment Analyzer")
    st.write(
        'Upload your file (either a PNG or PDF) or write some text. The app will predict its sentiment.')
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    uploaded_image = st.file_uploader("Upload a PNG file", type=['png'])
    user_input = st.text_input("Enter your text here")
    text = ":)"

    if uploaded_file is not None:
        # Convert PDF to text
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_image is not None:
        text = extract_text_from_img(uploaded_image)
    elif user_input:
        text = user_input

    # Sentiment analysis
    if text is not None:
        if st.button('Predict Sentiment'):
            sentiment_score = predict_sentiment(user_input)
            st.write('The sentiment score is:', sentiment_score["score"])
            st.write('The sentiment is:', sentiment_score["label"])

if selected_page == "Home":
    st.title("Home Page")
    st.write(
        "Welcome to VitaRize! This application is a customized document summarizer that you can use for anything!")
    st.write("This was made with OpenAI API and Langchain")
