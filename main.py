import openai
import streamlit as st
import PyPDF2
import pandas as pd
import altair as alt
import pytesseract as ps
import pyttsx3 as engine
import io
import os


from PIL import Image
from gtts import gTTS
from streamlit_option_menu import option_menu
from streamlit_chat import message
from transformers import pipeline
from googletrans import Translator, constants

# application created with open AI for summart, altair for pdf scraping, pytesseract for png scraping, gtts for text to speech
# and we implemented google translates API for translation

# pip install openai
# pip install streamlit
# pip install PyPDF2
# pip install transformers
# pip install torch
# pip install gtts
# pip install streamlit_option_menu
# pip install streamlit_chat
# pip install googletrans
# pip install pyttsx3
# brew install poppler-qt5
# pip install pytesseract / brew install tesseract (i would do both to be safe)
# once you get a new api key, go to secrets.toml and add it instead of pass

# run the code --> streamlit run main.py

# Set the GPT-3 API key
openai.api_key = st.secrets["pass"]

sentiment_score = {}

logo = "VitaRize.png"
st.image(logo, use_column_width=True)

selected_page = option_menu(
    menu_title=None,
    options=["Home", "Summarizer", "Sentiment",
             "Translator"],
    icons=["house", "info", "heart", "person-circle"],
    menu_icon="cast",
    default_index=3,
    orientation="horizontal",
)


def generate_answer(text, question):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Answer the following question based on these texts:\n\n{text}\n\nQuestion: {question} \n ",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    answer = response.choices[0].text.strip()
    return answer


def generate_summary(text, temperature, length):
    # Use GPT-3 to generate a summary of the article
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Summatize this paper " + text,
        temperature=temperature,
        max_tokens=length,
    )
    # return
    return response["choices"][0]["text"]


def extract_text_from_img(file):
    # Convert the image into a string
    text = str(((ps.image_to_string(Image.open(file)))))
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
    st.title("Summarizer")
    st.write('Upload your file (either a PNG or PDF), and you will get a summary!')

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

        if st.button("Generate Summary"):
            res = generate_summary(text, temperature, length)
            st.success(res)
            st.download_button('Download result', res)

            # Convert summary to speech
            tts = gTTS(text=res, lang='en', slow=False)
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            st.audio(audio_file, format='audio/mp3')

            question = st.text_input("Enter your question here")
            ans = generate_answer(text, question)
            st.success(ans)

    if uploaded_image is not None:
        # Convert PNG to text
        text = extract_text_from_img(uploaded_image)
        # Prompt settings
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        length = st.slider("Length of Summary", min_value=50,
                           max_value=500, step=50, value=100)

        if st.button("Generate Summary"):
            res = generate_summary(text, temperature, length)
            st.success(res)
            st.download_button('Download result', res)

            # Convert summary to speech
            tts = gTTS(text=res, lang='en', slow=False)
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            st.audio(audio_file, format='audio/mp3')

            question = st.text_input("Enter your question here")
            ans = generate_answer(text, question)
            st.success(ans)


if selected_page == "Sentiment":
    st.title("Sentiment")
    st.write(
        'Upload your file (either a PNG or PDF) or write some text and the app will predict its sentiment!')
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


if selected_page == "Translator":
    st.title("Translator")
    st.write(
        'Upload your file (either a PNG or PDF) and you will get a translation!')

    # Initialize the translator
    translator = Translator()

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

    if st.button("Translate"):
        if text:
            try:
                # Detect the language of the input text
                detected_language = translator.detect(text).lang
                st.write(f"Detected Language: {detected_language}")

                # Translate the text
                translated_text = translator.translate(
                    text, src=detected_language, dest='en')
                st.success(translated_text.text)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text to translate.")

if selected_page == "Questions":
    st.title("Questions")
    st.write(
        'Upload your file (either a PNG or PDF), and type in a question regarding the files you uploaded. The app will do its best to answer it.')
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    uploaded_image = st.file_uploader("Upload a PNG file", type=['png'])
    if uploaded_file is not None:
        # Convert PDF to text
        text = extract_text_from_pdf(uploaded_file)
        # Question settings
        question = st.text_input("Enter your question here")
        if st.button("Generate Answer"):
            st.write(generate_answer(text, question))

    if uploaded_image is not None:
        # Convert PNG to text
        text = extract_text_from_img(uploaded_image)
        # Prompt settings
        # Question settings
        question = st.text_input("Enter your question here")
        if st.button("Generate Answer"):
            st.write(generate_answer(text, question))


if selected_page == "Home":
    st.title("Home Page")
    st.write(
        "Welcome to VitaRize! This application is a customized document summarizer that you can use for anything! The application can summarize a file uploaded, predict its sentiment, translate the contents, and even answer questions about the file that you may have! You can navigate to any of the pages in the top bar to use the functions.")
    st.write("This was made with OpenAI API and Altair")
