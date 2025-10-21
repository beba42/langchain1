"""Entry point for launching the local LangChain agent."""

import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple
import gradio as gr
import os

from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_app.config.settings import settings
from langchain_app.agents.local_reader import chunk_documents, load_local_documents
from langchain_app.agents.web_search_agent import WebSearchAgent


def build_local_agent() -> Tuple[ConversationalRetrievalChain, ChatOllama]:
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

    print("[agent] Building FAISS (dense) index ...", flush=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    dense = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.3,
        },
    )

    print("[agent] Building BM25 (sparse) retriever ...", flush=True)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 8

    print("[agent] Combining dense + sparse into EnsembleRetriever ...", flush=True)
    retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.45, 0.55],
    )

    llm_model = settings.chat_model
    print(f"[agent] Spinning up ChatOllama model '{llm_model}' ...", flush=True)
    llm = ChatOllama(model=llm_model, temperature=0)

    print("[agent] Retrieval QA chain ready.", flush=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    return chain, llm

qa, llm = build_local_agent()
web_search_agent: WebSearchAgent | None = None
_web_search_error: str | None = None


def _ensure_web_search_agent() -> WebSearchAgent | None:
    """Instantiate the web search agent if needed, caching failures."""
    global web_search_agent, _web_search_error
    if web_search_agent is not None:
        return web_search_agent
    if _web_search_error is not None:
        return None
    try:
        web_search_agent = WebSearchAgent()
        return web_search_agent
    except Exception as exc:  # pragma: no cover - external dependency
        _web_search_error = str(exc)
        print(f"[agent] Web search unavailable: {exc}", flush=True)
        return None


if settings.search_enabled:
    _ensure_web_search_agent()

print("Local document QA agent ready.")

final_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant combining local knowledge with live web search. "
            "Use local findings when available, enrich with search insights, and note when you rely on the web.",
        ),
        (
            "human",
            "User question: {question}\n\n"
            "Conversation history:\n{history}\n\n"
            "Local retrieval answer:\n{local_answer}\n\n"
            "Local sources:\n{local_sources}\n\n"
            "Web search result:\n{web_search}\n\n"
            "Compose a clear, concise reply for the user.",
        ),
    ]
)

search_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's request into a focused web search query. Use keywords, expand abbreviations, "
            "and strip greetings or filler. Output only the search query text.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\n"
            "Current question:\n{question}\n\n"
            "Optimized search query:",
        ),
    ]
)


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


def _history_to_plaintext(history_messages: List[HumanMessage | AIMessage]) -> str:
    """Render LangChain chat history to plain text for prompting."""
    lines = []
    for msg in history_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _rewrite_for_search(question: str, history_text: str) -> str:
    """Use the main LLM to rewrite a question into a search-friendly query."""
    formatted = search_query_prompt.format_prompt(
        question=question,
        history=history_text or "(no prior messages)",
    )
    llm_reply = llm.invoke(formatted.to_messages())
    candidate = llm_reply.content if hasattr(llm_reply, "content") else str(llm_reply)
    return candidate.strip() or question


def respond(message, history, use_web_search: bool):
    """
    Gradio callback: given latest user message and full chat history, return model reply.
    """
    lc_history = lc_history_from_gradio(history)
    history_text = _history_to_plaintext(lc_history)
    qa_response = qa.invoke({"chat_history": lc_history, "question": message})
    local_answer = qa_response.get("answer") if isinstance(qa_response, dict) else qa_response
    sources = qa_response.get("source_documents", []) if isinstance(qa_response, dict) else []

    if not use_web_search:
        return local_answer  # already a string thanks to StrOutputParser

    web_result = ""
    search_agent = _ensure_web_search_agent()
    if search_agent is None:
        unavailable_reason = _web_search_error or "initialization failed"
        web_result = f"(web search unavailable: {unavailable_reason})"
    else:
        try:
            search_query = _rewrite_for_search(message, history_text)
            web_result = search_agent.run(search_query)
        except Exception as exc:  # pragma: no cover - network/tool failures are non-deterministic
            web_result = f"(web search unavailable: {exc})"

    source_text = "\n".join(getattr(doc, "page_content", str(doc)) for doc in sources)

    final = final_answer_prompt.format_prompt(
        question=message,
        history=history_text or "(no prior messages)",
        local_answer=local_answer,
        local_sources=source_text or "None",
        web_search=web_result or "No web results.",
    )
    llm_reply = llm.invoke(final.to_messages())
    return llm_reply.content if hasattr(llm_reply, "content") else str(llm_reply)

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
    default_search = bool(settings.search_enabled)

    def _toggle_search(current: bool) -> tuple[bool, str]:
        new_value = not current
        status = f"**Web search:** {'ON' if new_value else 'OFF'}"
        return new_value, status

    def _add_user_message(message: str, history: list[tuple[str, str | None]]) -> tuple[str, list[tuple[str, str | None]]]:
        if not message:
            return "", history
        updated = history + [(message, None)]
        return "", updated

    def _generate_bot_reply(history: list[tuple[str, str | None]], use_search: bool) -> list[tuple[str, str | None]]:
        if not history:
            return history
        question, _ = history[-1]
        prior = history[:-1]
        answer = respond(question, prior, use_search)
        history[-1] = (question, answer)
        return history

    with gr.Blocks(title="LangChain Chat", theme="default") as demo:
        gr.Markdown("# LangChain Chat\nA minimal chat UI powered by LangChain. Swap in your own chain.")

        search_state = gr.State(default_search)
        status_display = gr.Markdown(f"**Web search:** {'ON' if default_search else 'OFF'}")
        toggle_button = gr.Button("Toggle Web Search", variant="secondary")

        chatbot = gr.Chatbot(height=400, label="Conversation")
        with gr.Row():
            user_input = gr.Textbox(placeholder="Ask me anything...", lines=2, scale=8)
            send_button = gr.Button("Send", variant="primary", scale=1)
        clear_button = gr.Button("Clear Conversation", variant="secondary")
        gr.Examples(
            examples=["Summarize LangChain in 3 bullets.", "Give me a study plan for SQL basics."],
            inputs=user_input,
        )

        toggle_button.click(
            _toggle_search,
            inputs=search_state,
            outputs=[search_state, status_display],
            queue=False,
        )

        user_input.submit(
            _add_user_message,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(
            _generate_bot_reply,
            inputs=[chatbot, search_state],
            outputs=chatbot,
        )

        send_button.click(
            _add_user_message,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(
            _generate_bot_reply,
            inputs=[chatbot, search_state],
            outputs=chatbot,
        )

        clear_button.click(lambda: ([], ""), None, [chatbot, user_input], queue=False)

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

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
