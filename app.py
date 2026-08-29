import streamlit as st
from graph import run

st.set_page_config(page_title="Research Assistant", page_icon="🔍", layout="centered")
st.title("🔍 Research Assistant")
st.caption("Ask questions — I'll search your notes and/or the web as needed")

st.divider()

question = st.text_input(
    "Ask a question:",
    placeholder="What is machine learning? What's the latest AI model?",
    key="question_input"
)

if question:
    with st.spinner("Researching..."):
        final_answer, route = run(question)
    
    st.markdown("### 💬 Answer")
    st.write(final_answer)
    
    st.divider()
    
    # Show routing decision
    route_display = {
        "web": "🌐 Web Search",
        "notes": "📓 Personal Notes",
        "both": "🌐 Web + 📓 Notes"
    }
    
    st.markdown(f"**Research method:** {route_display.get(route, route)}")