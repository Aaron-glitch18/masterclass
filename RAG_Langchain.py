import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

def main():
    load_dotenv()

    st.set_page_config(page_title="CivilStat AI")
    st.header("CivilStat AI")

    # Initialisation mémoire
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pdf = st.file_uploader("Importez votre PDF", type="pdf")

    if pdf is not None:

        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.warning("Impossible d'extraire du texte de ce PDF (scan image ?).")
            return

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)

        with st.spinner("Indexation du document..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            knowledge_base = FAISS.from_texts(chunks, embedding=embeddings)

        st.success(f"Document indexé ({len(chunks)} segments)")

        question = st.text_input("Posez une question sur ce document")

        if question:
            groq_api_key = os.getenv("GROQ_API_KEY") or "votre_cle_dev"

            llm = ChatGroq(
                api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile"
            )

            # Recherche des chunks pertinents
            docs = knowledge_base.similarity_search(question, k=4)
            top_chunks = [doc.page_content for doc in docs]

            #Construction de l'historique
            history_text = ""
            for q, r in st.session_state.chat_history:
                history_text += f"Question: {q}\nRéponse: {r}\n"

            #  Prompt avec mémoire
            prompt = f"""
Tu es un assistant intelligent.

Utilise le CONTEXTE et l'HISTORIQUE pour répondre.

Si la réponse n'est pas dans le document, dis : "Je ne sais pas".

HISTORIQUE :
{history_text}

CONTEXTE :
{chr(10).join(top_chunks)}

QUESTION :
{question}

RÉPONSE :
"""

            with st.spinner("Recherche de la réponse..."):
                response = llm.invoke(prompt)

            response_text = response.content

            st.session_state.chat_history.append((question, response_text))

            #  Limiter la mémoire (optionnel)
            st.session_state.chat_history = st.session_state.chat_history[-5:]

            st.markdown("### Réponse")
            st.write(response_text)

        #  Affichage conversation
        if st.session_state.chat_history:
            st.markdown("###  Historique des échanges")

            for q, r in st.session_state.chat_history:
                st.markdown(f"** Question :** {q}")
                st.markdown(f"** Réponse :** {r}")
                st.markdown("---")

if __name__ == "__main__":
    main()