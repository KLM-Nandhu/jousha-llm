import streamlit as st
import openai
import chardet
import PyPDF2
import docx

# Load your API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

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

def split_text(text, max_tokens=2000):
    """ Split the text into smaller chunks to avoid exceeding token limits. """
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        current_tokens += 1
        current_chunk.append(word)
        if current_tokens >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def ask_question_to_llm(content_chunks, question):
    """ Sends each chunk of content and the user's question to the LLM and gets a response. """
    all_responses = []
    total_tokens = 0

    for chunk in content_chunks:
        prompt = f"Based on the following content, answer the user's question using only the information from the document:\n\nContent:\n{chunk}\n\nQuestion: {question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        response_text = response['choices'][0]['message']['content'].strip()
        all_responses.append(response_text)
        total_tokens += response['usage']['total_tokens']

    return " ".join(all_responses), total_tokens

def main():
    st.title('LLM Interaction')
    st.write("Upload multiple documents, and then ask a question based on their content.")

    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)  # Accept text, PDF, and Word files
    if uploaded_files:
        combined_content = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "text/plain":
                raw_data = uploaded_file.getvalue()
                result = chardet.detect(raw_data)
                file_encoding = result['encoding']
                file_content = raw_data.decode(file_encoding)
            elif uploaded_file.type == "application/pdf":
                file_content = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_content = read_docx(uploaded_file)
            
            combined_content += file_content + "\n"

        if combined_content:
            content_chunks = split_text(combined_content)
            
            user_question = st.text_input("Ask a question based on the documents' content:")
            if user_question:
                response_text, token_usage = ask_question_to_llm(content_chunks, user_question)
                # Display the LLM's response and token usage
                st.write("### LLM's Response")
                st.text_area("Here's the response from the LLM:", response_text, height=300)
                st.write(f"Tokens used in this request: {token_usage}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
