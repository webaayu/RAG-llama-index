import streamlit as st
import fitz  # PyMuPDF
import zipfile
import io
import os
from llama_index import SimpleIndex
from langchain.llms import Ollama
from bs4 import BeautifulSoup

# Function to extract text from PDF file
def extract_text_from_pdf(file):
    try:
        with fitz.open(stream=file, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except fitz.fitz.PDFError as e:
        print(f"Error occurred while processing PDF: {e}")
        return ""

# Function to extract text from HTML file
def extract_text_from_html(file):
    try:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error occurred while processing HTML: {e}")
        return ""

# Function to extract text from text file
def extract_text_from_txt(file):
    try:
        return file.decode("utf-8")
    except Exception as e:
        print(f"Error occurred while processing text file: {e}")
        return ""

# Get response from LLM
def get_llm_response(input, content, prompt):
    # Loading Ollama Llama2 model
    model = Ollama(model='llama2')
    cont = str(content)
    response = model.invoke([input, cont, prompt])  # Get response from model
    return response

# Main function
def main():
    # Set title and description
    st.title("ZIP File Chatbot with Llama-Index")

    # Create a sidebar for file upload
    st.sidebar.title("Upload ZIP File")
    uploaded_file = st.sidebar.file_uploader("Choose a ZIP file", type=['zip'])

    if uploaded_file is not None:
        # Read the uploaded file as a byte stream
        bytes_data = uploaded_file.read()
        zip_file = io.BytesIO(bytes_data)

        # Extract ZIP file contents
        extracted_texts = []
        with zipfile.ZipFile(zip_file, 'r') as z:
            for file_info in z.infolist():
                with z.open(file_info) as file:
                    if file_info.filename.endswith('.pdf'):
                        pdf_text = extract_text_from_pdf(file.read())
                        if pdf_text:
                            extracted_texts.append(pdf_text)
                    elif file_info.filename.endswith('.html') or file_info.filename.endswith('.htm'):
                        html_text = extract_text_from_html(file.read())
                        if html_text:
                            extracted_texts.append(html_text)
                    elif file_info.filename.endswith('.txt'):
                        txt_text = extract_text_from_txt(file.read())
                        if txt_text:
                            extracted_texts.append(txt_text)

        # Combine extracted texts
        combined_text = "\n".join(extracted_texts)
        if combined_text:
            try:
                # Create llama-index
                index = SimpleIndex()
                index.add_texts([combined_text])

                st.write("Texts indexed successfully with Llama-Index.")

            except Exception as e:
                st.error(f"Error occurred during text processing: {e}")

    # Text input for prompt
    prompt = st.text_input("Ask a Question", "")

    # Submit button
    submitted = st.button("Submit")

    if submitted:
        if prompt:
            results = index.query(prompt)
            if results:
                text = results[0]['text']
                input_prompt = """You are an expert in understanding text contents. You will receive input files and you will have to answer questions based on the input files."""
                response = get_llm_response(input_prompt, text, prompt)
                st.subheader("Generated Answer:")
                st.write(response)
            else:
                st.write("No relevant documents found.")

if __name__ == "__main__":
    main()