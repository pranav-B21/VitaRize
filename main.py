import openai
import streamlit as st
from streamlit_option_menu import option_menu
import PyPDF2

# pip install openai
# pip install streamlit
# pip install PyPDF2
# once you get a new api key, go to secrets.toml and add it instead of pass

# run the code --> streamlit run main.py

# Set the GPT-3 API key
openai.api_key = st.secrets["pass"]

logo = "Design.png"
st.image(logo, use_column_width=True)

selected_page = option_menu(
    menu_title=None,
    options=["Home", "Summarizer"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


def generate_summary(text, temperature, length):
    # Use GPT-3 to generate a summary of the article
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="This is the user prompt for how the text should be summarized" +
        "here is the text:" + text,
        temperature=temperature,
        max_tokens=length,
    )
    # Print the generated summary
    res = response["choices"][0]["text"]
    st.success(res)
    st.download_button('Download result', res)


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


if selected_page == "Summarizer":
    st.title("PDF Summarization")
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Convert PDF to text
        text = extract_text_from_pdf(uploaded_file)

        # Prompt settings
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        length = st.slider("Length of Summary", min_value=50,
                           max_value=500, step=50, value=100)

        if st.button("Generate Summary"):
            summary = generate_summary(text, temperature, length)
            st.header("Summary")
            st.write(summary)

if selected_page == "Home":
    st.title("Home Page")
    st.write(
        "Welcome to VitaRize! This application is a customized document summarizer that you can use for anything!")
    st.write("This was made with OpenAI API and Langchain")
