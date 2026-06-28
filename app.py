"""
Zyro Dynamics HR Help Desk — RAG Chatbot
Streamlit app for the NxtWave RAG Challenge.

Deploy on https://share.streamlit.io :
  1. Push this file + requirements.txt + a `data/` folder containing the
     11 Zyro Dynamics HR PDFs to a public GitHub repo.
  2. On Streamlit Cloud, set the app's "Secrets" to:
        GROQ_API_KEY = "your_groq_key"
        LANGCHAIN_API_KEY = "your_langsmith_key"
  3. Deploy. First load will take ~30-60s to build the vector index
     (cached after that).
"""

import os
import glob

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🧑‍💼",
    layout="centered",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LLM_MODEL = "llama-3.3-70b-versatile"

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics' policy "
    "documents (leave, WFH, code of conduct, performance, compensation, "
    "IT/data security, POSH, onboarding/separation, and travel & expense). "
    "This question falls outside that scope, so I'm not able to help with "
    "it here."
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are the HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Answer the employee's question using ONLY the context below, which is taken
directly from Zyro Dynamics' official HR policy documents. Do not use any
outside knowledge. If the context does not contain enough information to
answer confidently, say you don't have that information in the policy
documents instead of guessing.

Keep the answer concise, accurate, and easy to read for an employee.

Context:
{context}

Question:
{question}

Answer:"""
)

OOS_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict scope classifier for an HR Help Desk bot.

Zyro Dynamics' HR policy documents cover ONLY these topics: company profile,
employee handbook, leave policy (EL/SL/maternity/paternity), work from home
policy, code of conduct, performance review policy (APR/PIP/ratings),
compensation & benefits policy, IT & data security policy, prevention of
sexual harassment (POSH) policy, onboarding & separation policy, and travel
& expense policy.

Decide whether the question below could plausibly be answered using ONLY
those documents.

Question: {question}

Respond with exactly one word, nothing else: "IN_SCOPE" or "OUT_OF_SCOPE"."""
)


# --------------------------------------------------------------------------
# Cached setup: build the index once, reuse across user sessions
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HR policies and building the index…")
def load_retriever():
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        st.error(
            f"No PDFs found in `{DATA_DIR}`. Make sure the 11 Zyro Dynamics "
            "HR policy PDFs are committed to the `data/` folder of this repo."
        )
        st.stop()

    documents = []
    for path in pdf_paths:
        documents.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )
    return retriever


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def ask_bot(question: str, retriever, llm):
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()
    verdict = classifier_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = sorted({
        os.path.basename(d.metadata.get("source", "unknown")) for d in docs
    })
    return {"answer": answer, "sources": sources}


# --------------------------------------------------------------------------
# Environment / secrets
# --------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not set. Add it under your Streamlit app's "
        "**Settings → Secrets** before chatting.",
        icon="⚠️",
    )

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")
st.caption(
    "Ask me about leave, work from home, code of conduct, performance "
    "reviews, compensation, IT & data security, POSH, onboarding/"
    "separation, or travel & expense policies."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm the Zyro Dynamics HR assistant. What HR question can I help with today?",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

user_question = st.chat_input("Type your HR question…")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            answer = "I can't reach the LLM right now — GROQ_API_KEY is missing. Please contact the app owner."
            sources = []
        else:
            with st.spinner("Checking the policy documents…"):
                retriever = load_retriever()
                llm = load_llm()
                result = ask_bot(user_question, retriever, llm)
                answer = result["answer"]
                sources = result["sources"]

        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )