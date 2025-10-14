"""Entry point for launching the local LangChain agent."""

import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple
import gradio as gr
import os

from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_app.config.settings import settings
from langchain_app.agents.local_reader import chunk_documents, load_local_documents


def build_local_agent() -> ConversationalRetrievalChain:
    """Construct a RetrievalQA agent backed by local documents."""
    data_dir = settings.data_dir
    print(f"[agent] Loading local documents from {data_dir} ...", flush=True)
    documents = load_local_documents()
    if not documents:
        raise RuntimeError("No local documents found in the data directory.")

    print(f"[agent] Loaded {len(documents)} document(s). Splitting into chunks ...", flush=True)
    chunks = chunk_documents(documents)
    print(f"[agent] Created {len(chunks)} chunk(s). Initializing embeddings ...", flush=True)

    embeddings_model = settings.embedding_model
    print(f"[agent] Connecting to Ollama embeddings model '{embeddings_model}' (this may take a moment) ...", flush=True)
    embeddings = OllamaEmbeddings(model=embeddings_model)
    print("[agent] Building FAISS vector store ...", flush=True)
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever()

    llm_model = settings.chat_model
    print(f"[agent] Spinning up ChatOllama model '{llm_model}' ...", flush=True)
    llm = ChatOllama(model=llm_model, temperature=0)

    print("[agent] Retrieval QA chain ready.", flush=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

qa = build_local_agent()

print("Local document QA agent ready.")

def lc_history_from_gradio(history):
    """
    Convert Gradio's history format [(user, bot), ...] -> LangChain [HumanMessage, AIMessage, ...]
    """
    msgs = []
    for user_msg, bot_msg in history:
        if user_msg:
            msgs.append(HumanMessage(content=user_msg))
        if bot_msg:
            msgs.append(AIMessage(content=bot_msg))
    return msgs

def respond(message, history):
    """
    Gradio callback: given latest user message and full chat history, return model reply.
    """
    lc_history = lc_history_from_gradio(history)
    reply = qa.invoke({"chat_history": lc_history, "question": message})['answer']
    return reply  # already a string thanks to StrOutputParser

#def _spinner(message: str, stop_event: threading.Event) -> None:
#    """Display a simple spinner while the model is thinking."""
#    frames = "|/-\\"
#    idx = 0
#    sys.stdout.write(f"{message} ")
#    sys.stdout.flush()
#    while not stop_event.is_set():
#        sys.stdout.write(frames[idx % len(frames)])
#        sys.stdout.flush()
#        time.sleep(0.1)
#        sys.stdout.write("\b")
#        idx += 1
#    sys.stdout.write("done\n")
#    sys.stdout.flush()


def main() -> None:
    """Launch an interactive query loop."""
    

    gr.ChatInterface(
        fn=respond,
        title="LangChain Chat",
        description="A minimal chat UI powered by LangChain. Swap in your own chain.",
        textbox=gr.Textbox(placeholder="Ask me anything...", lines=2),
        examples=["Summarize LangChain in 3 bullets.", "Give me a study plan for SQL basics."],
        theme="default",
    ).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

#    try:
#        history: List[Tuple[str, str]] = []
#        while True:
#
#            question = input("You> ").strip()
#            if question.lower() in {"exit", "quit"}:
#                break
#            if not question:
#                continue
#
#            thinking_event = threading.Event()
#            spinner_thread = threading.Thread(target=_spinner, args=("[agent] Thinking", thinking_event), daemon=True)
#            spinner_thread.start()
#            response = qa.invoke({"question": question, "chat_history": history})
#            thinking_event.set()
#            spinner_thread.join()
#
#            answer = response.get("answer") if isinstance(response, dict) else response
#            history.append((question, answer))
#            print(f"Agent> {answer}")
#    except KeyboardInterrupt:
#        print("\nInterrupted by user. Shutting down agent.")


if __name__ == "__main__":
    main()
