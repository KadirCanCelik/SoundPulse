import json
import logging
import os
import time

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ANALYZE_URL = f"{BASE_URL}/api/v1/analyze"
QA_URL = f"{BASE_URL}/api/v1/qa"
STATUS_BASE_URL = f"{BASE_URL}/api/v1/status"

st.set_page_config(
    page_title="SoundPulse",
    layout="wide"
)

def config_ui():
    """Applies custom CSS styling to the Streamlit UI."""
    st.markdown("""
    <style>
    footer {visibility: hidden;}

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-left: 4px solid #2563EB;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    [data-testid="stSegmentedControl"] {
        padding-bottom: 10px;
    }

    [data-testid="stFileUploadDropzone"] {
        border-radius: 12px;
        border: 2px dashed #94A3B8;
        background-color: #F1F5F9;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #2563EB;
        background-color: #EFF6FF;
    }
    </style>
    """, unsafe_allow_html=True)

config_ui()

st.title("SoundPulse: Audio Intelligence")
st.markdown("Upload your meeting audio to generate transcripts, executive summaries, and interrogate your meeting data using a strictly grounded, RAG-powered QA agent.")

if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

with st.sidebar:
    st.header("Upload & Configuration")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac"]
    )

    num_speakers = st.slider(
        "Expected Number of Speakers",
        min_value=1,
        max_value=10,
        value=2,
        help="Helps the AI cluster speaker voices accurately."
    )

    analyze_button = st.button("Analyze Audio", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# ASYNCHRONOUS POLLING LOGIC
# -----------------------------------------------------------------------------

if analyze_button:

    if uploaded_file is None:
        st.sidebar.error("Please upload an audio file")

    else:
        with st.spinner("Dispatching audio to the processing queue..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                request_payload = {"num_speakers": num_speakers}

                response = requests.post(ANALYZE_URL, files=files, data=request_payload, timeout=30)
                
                if response.status_code == 200:
                    job_data = response.json()
                    job_id = job_data.get("job_id")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    job_id = None
            except Exception as e:
                st.error(f"Connection error: {e}")
                job_id = None

        if job_id:
            with st.status("Task queued. Waiting for GPU Worker...", expanded=True) as status_box:
                
                while True:
                    try:
                        status_res = requests.get(f"{BASE_URL}/api/v1/status/{job_id}", timeout=10)
                        if status_res.status_code != 200:
                            status_box.update(label="Status check failed!", state="error")
                            break
                            
                        result_data = status_res.json()
                        current_status = result_data.get("status")
                        
                        if current_status == "SUCCESS":
                            status_box.update(label="Analysis completely successfully!", state="complete", expanded=False)
                            worker_result = result_data.get("result", {})
                            
                            # Retrieve the heavy JSON payloads directly from the shared Docker volume
                            with open(worker_result["transcript_path"], "r", encoding="utf-8") as f:
                                transcript_json = json.load(f)
                                
                            with open(worker_result["analysis_path"], "r", encoding="utf-8") as f:
                                analysis_json = json.load(f)

                            # Populate session state for UI rendering
                            st.session_state["analysis_data"] = {
                                "meeting_id": worker_result["meeting_id"],
                                "transcript": transcript_json,
                                "analysis": analysis_json,
                                "pdf_filepath": worker_result.get("pdf_filepath")
                            }
                            st.session_state["chat_messages"] = [
                                {"role": "assistant", "content": "Hello! I've successfully analyzed the meeting. What would you like to know?"}
                            ]
                            
                            # Trigger a rerun to display the lower UI section
                            st.rerun()
                            break
                            
                        elif current_status == "FAILED":
                            error_msg = result_data.get("result", "Unknown error")
                            status_box.update(label=f"Task failed: {error_msg}", state="error")
                            break
                            
                        elif current_status == "PROCESSING":
                            meta = result_data.get("meta") or {}
                            step_msg = meta.get("step", "Processing...")
                            progress_val = meta.get("progress", 0)
                            status_box.update(label=f"{step_msg} ({progress_val}%)")
                            
                        else:
                            status_box.update(label="Waiting in queue (PENDING)...")
                            
                    except Exception as e:
                        status_box.update(label=f"Polling error: {e}", state="error")
                        break
                    
                    # Wait before asking API again
                    time.sleep(3)       


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

if st.session_state["analysis_data"]:
    meeting_data = st.session_state["analysis_data"]
    meeting_id = meeting_data["meeting_id"]
    analysis = meeting_data["analysis"]
    transcript = meeting_data["transcript"]
    pdf_path = meeting_data.get("pdf_filepath")

    selected_view = st.segmented_control("Navigation", ["Overview", "Transcript", "Ask QA Agent"], default="Overview", label_visibility="collapsed")

    if not selected_view:
        selected_view = "Overview"
    
    st.markdown("---")
    
    if selected_view == "Overview":
        if pdf_path:
            pdf_filename = os.path.basename(pdf_path)
            clean_display_path = f"soundpulse/exports/{pdf_filename}"
            st.success(f"Your detailed meeting report has been saved on your machine: {clean_display_path}")
            st.markdown("---")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Executive Summary")
            st.info(analysis.get("executive_summary", "No summary available."))
            
            st.subheader("Key Decisions")
            if analysis.get("key_decisions"):
                for dec in analysis["key_decisions"]:
                    st.markdown(f"**{dec['topic']}**: {dec['decision_summary']} *(Ref: {dec['start_time']}s)*")
            else:
                st.write("No key decisions recorded.")
                
        with col2:
            st.subheader("Sentiment")
            sentiment = analysis.get("sentiment", "Neutral")
            st.metric(label="Overall Tone", value=sentiment)
            
            st.subheader("Action Items")
            if analysis.get("action_items"):
                for item in analysis["action_items"]:
                    st.markdown(f"- **[{item['assignee']}]** {item['task_description']} *(Ref: {item['start_time']}s)*")
            else:
                st.write("No action items assigned.")

    elif selected_view == "Transcript":
        st.subheader("Meeting Transcript")
        with st.container(height=500):
            for seg in transcript:
                st.markdown(f"**[{seg['start']:05.1f} - {seg['end']:05.1f}] {seg['speaker']}:** {seg['text']}")

    elif selected_view == "Ask QA Agent":
        st.subheader("Chat with your Meeting Data")
        
        chat_container = st.container(height=500)
        
        with chat_container:
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask a question about this meeting..."):
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):
                        real_history = [
                            msg for msg in st.session_state["chat_messages"][:-1]
                            if msg["content"] != "Hello! I've successfully analyzed the meeting. What would you like to know?"
                        ]

                        qa_payload = {
                            "meeting_id": meeting_id,
                            "question": prompt,
                            "chat_history": real_history
                        }
                        try:
                            qa_response = requests.post(QA_URL, json=qa_payload, timeout=60)
                            if qa_response.status_code == 200:
                                answer = qa_response.json().get("answer", "No answer returned.")
                                st.markdown(answer)
                                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                            else:
                                st.error(f"QA Agent Error: {qa_response.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")