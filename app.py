import streamlit as st
import openai
import chardet
import PyPDF2
import docx
import pypandoc
import pandas as pd

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Initialize the session state to store total tokens used
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def read_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def read_rtf(file):
    try:
        output = pypandoc.convert_text(file.read().decode('utf-8'), 'plain', format='rtf')
        return output
    except Exception as e:
        st.error(f"Error reading RTF file: {e}")
        return ""

def read_metadata(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading metadata file: {e}")
        return None

def split_text(text, max_tokens=2000):
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

def ask_question_to_llm(prompt, max_tokens=150):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    response_text = response['choices'][0]['message']['content'].strip()
    total_tokens = response['usage']['total_tokens']
    return response_text, total_tokens

def find_relevant_files(metadata_content, user_question):
    relevant_files = set()
    if not isinstance(metadata_content, pd.DataFrame):
        st.error("Metadata should be in a tabular format.")
        return relevant_files

    for index, row in metadata_content.iterrows():
        if 'Metadata tags' in row and any(tag.lower() in user_question.lower() for tag in row['Metadata tags'].split(',')):
            relevant_files.add(row['File Name'])
    
    return relevant_files

def main():
    st.title('LLM Interaction')
    st.write("Upload multiple documents, and then ask a question based on their content.")

    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "docx", "rtf", "doc"], accept_multiple_files=True)
    metadata_file = st.file_uploader("Upload metadata file", type=["xlsx"])  # Only allow Excel files for metadata

    if uploaded_files and metadata_file:
        file_contents = {}

        metadata_content = read_metadata(metadata_file)
        if metadata_content is None:
            st.write("Unsupported metadata file format.")
            return
        
        for uploaded_file in uploaded_files:
            file_content = ""
            try:
                if uploaded_file.type == "text/plain":
                    raw_data = uploaded_file.getvalue()
                    result = chardet.detect(raw_data)
                    file_encoding = result['encoding']
                    file_content = raw_data.decode(file_encoding)
                elif uploaded_file.type == "application/pdf":
                    file_content = read_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.type == "application/msword":
                    file_content = read_docx(uploaded_file)
                elif uploaded_file.type == "application/rtf":
                    file_content = read_rtf(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file {uploaded_file.name}: {e}")
                continue
            
            file_contents[uploaded_file.name] = file_content
        
        user_question = st.text_input("Ask a question based on the documents' content:")
        if user_question:
            relevant_files = find_relevant_files(metadata_content, user_question)
            
            st.write("Relevant files based on metadata:", list(relevant_files))
            
            relevant_chunks = []
            chunk_details = []
            for file_name, content in file_contents.items():
                if file_name in relevant_files:
                    chunks = split_text(content)
                    relevant_chunks.extend(chunks)
                    chunk_details.extend([(file_name, i, chunk) for i, chunk in enumerate(chunks)])
                    st.write(f"Number of chunks for {file_name}: {len(chunks)}")

            if chunk_details:
                st.write("### Chunks")
                for file_name, i, chunk in chunk_details:
                    if st.button(f"Show details for {file_name} - Chunk {i+1}"):
                        st.write(f"### {file_name} - Chunk {i+1}")
                        st.text_area("Chunk content", chunk, height=200)

            st.write("Number of relevant content chunks:", len(relevant_chunks))
            
            if relevant_chunks:
                all_responses = []
                total_token_usage = 0
                
                for file_name, i, chunk in chunk_details:
                    content_prompt = f"Based on the following content, answer the user's question using only the information from the document:\n\nContent:\n{chunk}\n\nQuestion: {user_question}"
                    response_text, token_usage = ask_question_to_llm(content_prompt)
                    all_responses.append((file_name, i, response_text))
                    total_token_usage += token_usage
                
                combined_response = "\n".join([f"From {file_name}, Chunk {i+1}:\n{response}\n" for file_name, i, response in all_responses])
                
                st.session_state.total_tokens_used += total_token_usage
                
                st.write("### LLM's Response")
                st.text_area("Here's the response from the LLM:", combined_response, height=300)
                st.write(f"Tokens used in this request: {total_token_usage}")
                st.write(f"Total tokens used: {st.session_state.total_tokens_used}")
            else:
                st.write("No relevant content found to answer the question.")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
