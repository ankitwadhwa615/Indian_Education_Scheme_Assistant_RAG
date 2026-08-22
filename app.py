import time
from pathlib import Path

import streamlit as st

from rag import get_answer

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Indian Education Scheme Assistant",
    page_icon="assets/indian_education_rag_logo.png",
    layout="wide"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>
.stMarkdown p {
    font-size: 18px !important;
}

h1 {
    font-size: 3rem !important;
}

.stChatMessage {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# SESSION STATE
# ==========================

if "chats" not in st.session_state:
    st.session_state.chats = {
        "New Chat": []
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

# Safety checks

if len(st.session_state.chats) == 0:
    st.session_state.chats["New Chat"] = []

if (
    st.session_state.current_chat
    not in st.session_state.chats
):
    st.session_state.current_chat = list(
        st.session_state.chats.keys()
    )[0]

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("💬 Chats")

# Create New Chat

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    chat_name = "New Chat"

    counter = 1

    while chat_name in st.session_state.chats:
        chat_name = f"New Chat {counter}"
        counter += 1

    st.session_state.chats[chat_name] = []
    st.session_state.current_chat = chat_name

    st.rerun()

# Existing Chats

for chat_name in list(st.session_state.chats.keys()):

    col1, col2 = st.sidebar.columns([5, 1])

    with col1:

        if st.button(
            chat_name,
            key=f"open_{chat_name}",
            use_container_width=True
        ):
            st.session_state.current_chat = chat_name
            st.rerun()

    with col2:

        if st.button(
            "🗑️",
            key=f"delete_{chat_name}"
        ):

            del st.session_state.chats[chat_name]

            if len(st.session_state.chats) == 0:

                st.session_state.chats["New Chat"] = []

                st.session_state.current_chat = "New Chat"

            elif (
                st.session_state.current_chat
                == chat_name
            ):

                st.session_state.current_chat = list(
                    st.session_state.chats.keys()
                )[0]

            st.rerun()

st.sidebar.markdown("---")

st.sidebar.caption("Answers are generated from the local scheme database. Verify eligibility and deadlines on official portals.")

st.sidebar.markdown(
    f"**Current Chat:**\n\n{st.session_state.current_chat}"
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 🚀 Tech Stack

- LangChain
- ChromaDB
- BAAI BGE Embeddings
- Groq API
- Openai/gpt-oss-120b
- Streamlit
""")

# ==========================
# MAIN TITLE
# ==========================

logo_path = Path(__file__).parent / "assets" / "india_education_logo.png"
logo_col, title_col = st.columns([1, 7], vertical_alignment="center")


with title_col:
    st.title("Indian Education Scheme Assistant")

# ==========================
# DISPLAY CHAT
# ==========================

messages = st.session_state.chats[
    st.session_state.current_chat
]

for msg in messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def format_history(chat_messages, limit=8):
    """Keep follow-up questions grounded without sending the full chat indefinitely."""
    return "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in chat_messages[-limit:]
    )

# ==========================
# CHAT INPUT
# ==========================

query = st.chat_input(
    "Ask about government schemes..."
)

# ==========================
# PROCESS QUERY
# ==========================

if query:

    messages = st.session_state.chats[
        st.session_state.current_chat
    ]

    # Auto rename first message

    if (
        st.session_state.current_chat.startswith("New Chat")
        and len(messages) == 0
    ):

        new_title = query[:40]

        counter = 1
        base_title = new_title

        while new_title in st.session_state.chats:

            new_title = (
                f"{base_title} ({counter})"
            )

            counter += 1

        st.session_state.chats[new_title] = messages

        del st.session_state.chats[
            st.session_state.current_chat
        ]

        st.session_state.current_chat = new_title

        messages = st.session_state.chats[new_title]

    # User message

    messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    start = time.time()

    try:
        with st.spinner("Finding relevant government schemes..."):
            result = get_answer(query, history=format_history(messages[:-1]))
    except (FileNotFoundError, RuntimeError) as error:
        result = {"answer": f"I’m unable to answer that right now. {error}", "sources": []}
    except Exception:
        result = {
            "answer": "I ran into an unexpected error while searching the scheme database. Please try again shortly.",
            "sources": [],
        }

    elapsed = time.time() - start

    answer = result["answer"]

    messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander(f"📚 Sources ({len(result['sources'])})"):

        for index, source in enumerate(result["sources"], start=1):

            st.markdown(f"### {index}. {source.metadata.get('scheme_name', 'Unknown scheme')}")

            st.write(
                source.page_content[:500]
            )

    with st.expander("🔍 Retrieved Chunks"):

        for i, doc in enumerate(result["sources"]):

            st.write(f"Chunk {i+1}")

            st.code(
                doc.page_content[:1000]
            )

    st.caption(
        f"⏱ Response Time: {elapsed:.2f}s"
    )
