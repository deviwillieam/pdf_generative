from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 😎")

    with st.sidebar:
        st.image("itb_black.png", width=200)
        # default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        # Define the correct password
        api_key = os.getenv("OPENAI_API_KEY")
        correct_password = os.getenv("CORRECT_PASSWORD")

        # Prompt the user for the password
        password_input = st.text_input("Enter Password to Access AI Settings", type="password")

        # Check if the entered password matches the correct password
        if password_input == correct_password:
            with st.expander("🔐 AI Settings"):
                with st.container():
                    openai_api_key = st.text_input("Input API (https://platform.openai.com/)", value=api_key,
                                                   type="password")
        else:
            st.warning("Incorrect password. Please try again.")
        with st.popover("✨ Model"):
            model = st.selectbox("Select a model:", [
                "gpt-4o-2024-05-13",
		"gpt-4o-mini-2024-07-18",
		"gpt-4-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
            ], index=1)

        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")


    else:
        client = OpenAI(api_key=openai_api_key)
        api_key = os.getenv("OPENAI_API_KEY")
        pdf = st.file_uploader("Upload your PDF", type="pdf")

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = st.text_input("Ask a question about your PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)

                st.write(response)


if __name__ == '__main__':
    main()