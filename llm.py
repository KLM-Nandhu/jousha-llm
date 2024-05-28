import streamlit as st
import openai
import chardet
import PyPDF2
import docx
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def summarize_content(content):
    """ Summarizes the content if it exceeds a certain length. """
    if len(content.split()) > 2000:  # Arbitrary limit for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please summarize the following content:\n\n{content}"}
            ],
            max_tokens=1000
        )
        summarized_content = response['choices'][0]['message']['content'].strip()
        return summarized_content
    else:
        return content

def ask_question_to_llm(content, question):
    """ Sends summarized content and user's question to an LLM and gets a response. """
    prompt = f"Based on the following content, answer the user's question:\n\nContent:\n{content}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def main():
    st.title('LLM Interaction')
    st.write("Upload any document, and then ask a question based on its content.")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])  # Accept text, PDF, and Word files
    if uploaded_file is not None:
        file_content = ""
        if uploaded_file.type == "text/plain":
            raw_data = uploaded_file.getvalue()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']
            file_content = raw_data.decode(file_encoding)
        elif uploaded_file.type == "application/pdf":
            file_content = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_content = read_docx(uploaded_file)
        
        if file_content:
            summarized_content = summarize_content(file_content)
            
            user_question = st.text_input("Ask a question based on the document content:")
            if user_question:
                response_text = ask_question_to_llm(summarized_content, user_question)
                # Display the LLM's response
                st.write("### LLM's Response")
                st.text_area("Here's the response from the LLM:", response_text, height=300)

# Run the Streamlit app
if __name__ == "__main__":
    main()
