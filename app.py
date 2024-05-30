import streamlit as st
import openai
import chardet
import PyPDF2
import docx
import pypandoc
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the session state to store total tokens used
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def read_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        # Silently log the error and return an empty string
        print(f"Error reading DOCX file: {e}")
        return ""

def read_rtf(file):
    output = pypandoc.convert_text(file.read().decode('utf-8'), 'plain', format='rtf')
    return output

def read_metadata(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    elif file.type == "text/plain":
        return file.getvalue().decode()
    elif file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    elif file.type == "application/msword":
        return read_docx(file)  # Use the same function as docx for compatibility
    elif file.type == "application/rtf":
        return read_rtf(file)
    else:
        return None

def split_text(text, max_tokens=1500):
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

def ask_question_to_llm(prompt, max_tokens=150):
    """ Sends the prompt to the LLM and gets a response. """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    response_text = response['choices'][0]['message']['content'].strip()
    total_tokens = response['usage']['total_tokens']
    return response_text, total_tokens

def generate_embeddings(text_list):
    """ Generate embeddings for a list of texts using OpenAI Embedding API. """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text_list
    )
    return np.array([embedding["embedding"] for embedding in response["data"]])

def find_relevant_files(metadata_content, user_question):
    relevant_files = set()
    if not isinstance(metadata_content, pd.DataFrame):
        st.error("Metadata should be in a tabular format.")
        return relevant_files

    # Extract metadata tags and filenames
    metadata_tags = metadata_content['Metadata tags'].tolist()
    file_names = metadata_content['File Name'].tolist()

    # Generate embeddings for metadata tags and the user question
    tag_embeddings = generate_embeddings(metadata_tags)
    question_embedding = generate_embeddings([user_question])[0]

    # Calculate cosine similarity between question and metadata tags
    similarities = cosine_similarity([question_embedding], tag_embeddings)[0]

    # Find the most relevant tags/files
    threshold = 0.5  # Adjust this threshold as needed
    for i, similarity in enumerate(similarities):
        if similarity > threshold:
            relevant_files.add(file_names[i])

    return relevant_files

def main():
    st.title('LLM Interaction')
    st.write("Upload multiple documents, and then ask a question based on their content.")

    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "docx", "rtf", "doc"], accept_multiple_files=True)  # Accept text, PDF, Word, and RTF files
    metadata_file = st.file_uploader("Upload metadata file", type=["xlsx"])  # Only allow Excel files for metadata

    if uploaded_files and metadata_file:
        file_contents = {}

        # Read metadata
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
                # Silently log the error and continue with the next file
                print(f"Error reading file {uploaded_file.name}: {e}")
                continue
            
            file_contents[uploaded_file.name] = file_content
            st.write(f"Content of {uploaded_file.name}:\n{file_content[:500]}...")  # Show the first 500 characters for debugging
        
        user_question = st.text_input("Ask a question based on the documents' content:")
        if user_question:
            # Step 1: Find relevant files based on metadata
            relevant_files = find_relevant_files(metadata_content, user_question)
            
            # Debug: Show which files are considered relevant
            st.write("Relevant files based on metadata:", list(relevant_files))
            
            # Step 2: Use LLM to process content of relevant files
            relevant_chunks = []
            for file_name, content in file_contents.items():
                if file_name in relevant_files:
                    st.write(f"Splitting content of {file_name} into chunks.")
                    chunks = split_text(content)
                    relevant_chunks.extend(chunks)
                    st.write(f"Number of chunks for {file_name}: {len(chunks)}")
            
            # Debug: Show number of relevant chunks
            st.write("Number of relevant content chunks:", len(relevant_chunks))
            
            if relevant_chunks:
                all_responses = []
                total_token_usage = 0  # Initialize token usage
                
                for chunk in relevant_chunks:
                    content_prompt = f"Based on the following content, answer the user's question using only the information from the document:\n\nContent:\n{chunk}\n\nQuestion: {user_question}"
                    response_text, token_usage = ask_question_to_llm(content_prompt)
                    all_responses.append(response_text)
                    total_token_usage += token_usage
                
                # Combine all responses into a single string
                combined_response = " ".join(all_responses)
                
                # Summarize the combined response
                summary_prompt = f"Summarize the following information to answer the user's question:\n\n{combined_response}"
                final_response, summary_token_usage = ask_question_to_llm(summary_prompt, max_tokens=300)
                
                # Update the session state with the tokens used
                st.session_state.total_tokens_used += total_token_usage + summary_token_usage
                
                st.write("### LLM's Response")
                st.text_area("Here's the summarized response from the LLM:", final_response, height=300)
                st.write(f"Tokens used in this request: {total_token_usage + summary_token_usage}")
                st.write(f"Total tokens used: {st.session_state.total_tokens_used}")
            else:
                st.write("No relevant content found to answer the question.")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
