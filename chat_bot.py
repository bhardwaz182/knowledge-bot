import os
import json
import re
import csv

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

# Load env vars at module level so cached functions can access them
load_dotenv()
os.environ.setdefault("USER_AGENT", "NLP_Project/1.0")

# --- Config ---
LLM_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_FILE = "cleaned_content_v2.txt"

# --- Core RAG functions ---

def split_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

@st.cache_resource(show_spinner=False)
def build_retriever():
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    docs = loader.load()
    doc_splits = split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = SKLearnVectorStore.from_documents(documents=doc_splits, embedding=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rewrite_query(question, chat_history=None):
    """Rewrite the user question into a keyword-rich search query for better retrieval."""
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in chat_history[-4:]  # last 2 turns
        )
    prompt = f"""Given the conversation history and a user question, rewrite the question as a keyword-rich search query that will best retrieve relevant documentation.

Rules:
- Include the core topic/feature name
- If the question asks "what is X" or "how does X work", also include terms like "configure", "setup", "settings" so both definition and configuration chunks are retrieved
- Keep it concise (under 15 words)

Conversation history:
{history_text}

User question: {question}

Return ONLY the rewritten search query, nothing else."""
    result = get_llm().invoke([HumanMessage(content=prompt)])
    return result.content.strip()

@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return ChatAnthropic(model=LLM_MODEL, temperature=0.0, api_key=api_key)

def generate_answer(question, context, chat_history=None):
    messages = [
        SystemMessage(content=f"""You are a helpful assistant for question-answering tasks about Surefire.
Use the following context to answer questions. If the answer is not in the context, say so.

Context:
{context}""")
    ]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    result = get_llm().invoke(messages)
    return result.content

# --- Hallucination grading ---

def extract_json(text):
    try:
        json_str = re.search(r'\{.*?\}', text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        return {'error': f'JSON extraction failed: {str(e)}', 'content': text}

def grade_hallucination(facts, generation):
    system_prompt = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Grade criteria:
(1) The STUDENT ANSWER must be grounded in the FACTS.
(2) The STUDENT ANSWER must not contain hallucinated information outside the FACTS.

Return ONLY valid JSON in this format:
{
    "score": <number between 0 and 1>,
    "explanation": "<your explanation>"
}"""

    user_prompt = f"FACTS:\n\n{facts}\n\nSTUDENT ANSWER: {generation}\n\nProvide only the JSON output."

    result = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return extract_json(result.content)

# --- Persistence ---

def save_results(question, answer, grading):
    data = {
        'question': question,
        'llm_model': LLM_MODEL,
        'embedding_model': EMBEDDING_MODEL,
        'answer': answer,
        'grading': grading,
    }
    with open('results.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

    csv_exists = os.path.isfile('results.csv')
    with open('results.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'llm_model', 'embedding_model', 'answer', 'score', 'explanation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({
            'question': question,
            'llm_model': LLM_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'answer': answer,
            'score': grading.get('score', ''),
            'explanation': grading.get('explanation', ''),
        })

# --- Streamlit UI ---

def main():
    st.title("Surefire Knowledge Bot")
    st.caption(f"Model: `{LLM_MODEL}` | Embeddings: `{EMBEDDING_MODEL}` | Source: `{KNOWLEDGE_FILE}`")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
        st.stop()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}
    if "langchain_history" not in st.session_state:
        st.session_state.langchain_history = []  # list of LangChain message objects

    # Sidebar controls
    with st.sidebar:
        st.header("Options")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.langchain_history = []
            st.rerun()
        show_context = st.checkbox("Show retrieved context", value=False)
        show_grading = st.checkbox("Show hallucination score", value=True)

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_grading and "grading" in msg:
                grading = msg["grading"]
                if "error" not in grading:
                    score = grading.get("score")
                    color = "green" if score is not None and score >= 0.7 else "orange" if score is not None and score >= 0.4 else "red"
                    st.caption(f"Hallucination score: :{color}[{score}] — {grading.get('explanation', '')}")
            if msg["role"] == "assistant" and show_context and "context" in msg:
                with st.expander("Retrieved context"):
                    if "search_query" in msg:
                        st.caption(f"Search query used: `{msg['search_query']}`")
                    st.text(msg["context"])

    # Chat input
    question = st.chat_input("Ask a question about Surefire...")
    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            retriever = build_retriever()
            search_query = rewrite_query(question, st.session_state.langchain_history)
            retrieved_docs = retriever.invoke(search_query)
            context = format_docs(retrieved_docs)
            answer = generate_answer(question, context, st.session_state.langchain_history)
            grading = grade_hallucination(context, answer)

        # Update LangChain history for next turn
        st.session_state.langchain_history.append(HumanMessage(content=question))
        from langchain_core.messages import AIMessage
        st.session_state.langchain_history.append(AIMessage(content=answer))

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)
            if show_grading and "error" not in grading:
                score = grading.get("score")
                color = "green" if score is not None and score >= 0.7 else "orange" if score is not None and score >= 0.4 else "red"
                st.caption(f"Hallucination score: :{color}[{score}] — {grading.get('explanation', '')}")
            if show_context:
                with st.expander("Retrieved context"):
                    st.caption(f"Search query used: `{search_query}`")
                    st.text(context)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "grading": grading,
            "context": context,
            "search_query": search_query,
        })

        save_results(question, answer, grading)

if __name__ == "__main__":
    main()
