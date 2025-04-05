import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import pytesseract
from PIL import Image
import pdfplumber
from mistralai import Mistral
import speech_recognition as sr
from gtts import gTTS
import base64
import os
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import re
from difflib import get_close_matches
import textwrap
import csv
import pandas as pd
import difflib

if "restoring_history" not in st.session_state:
    st.session_state["restoring_history"] = False

# Function to generate Google Calendar event link
def generate_google_calendar_link(title, start_datetime, end_datetime, details):
    #Creates a google calendar event URL
    base_url = "https://www.google.com/calendar/render?action=TEMPLATE"

    params = {
        "text": title,
        "dates": f"{start_datetime.replace('-', '').replace(':', '').replace(' ', 'T')}/{end_datetime.replace('-', '').replace(':', '').replace(' ', 'T')}",
        "details": details,
        
    }

    full_url = base_url + "&" + urllib.parse.urlencode(params)
    return full_url
#initializing the apis
vertexai.init(project="fluid-outcome-431318-i4", location="us-central1")
gemini_model = GenerativeModel("gemini-2.0-flash-001")
mistral_client = Mistral(api_key="T8D4i12helkPRJzwR3ZcySiP4LJg4AUX")

st.set_page_config(page_title="DocBot AI", page_icon="🩺", layout="wide")
st.sidebar.title("Select Mode")
mode = st.sidebar.radio("Choose an option", ["Medical Chatbot", "Insurance Coverage Check"])

#sidebar option
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["🌞 Light Mode", "🌙 Dark Mode"], index=0)
    st.divider()

    st.header("History")
    if "history" not in st.session_state:
        st.session_state["history"] = {}

    for title in st.session_state["history"]:
        if st.button(title):
            if not st.session_state["restoring_history"]:  # Prevent re-adding
                st.session_state["messages"] = st.session_state["history"][title].copy()
                st.session_state["restoring_history"] = True
                st.rerun()
    if st.button("🏠"):
        st.session_state["messages"] = []
        st.session_state["restoring_history"] = False
        st.rerun()

st.sidebar.header("Medicine Reminder Setup 🕰️ ")

title = st.sidebar.text_input("Reminder", "Take Crocin")
date = st.sidebar.date_input("Date", datetime.now() + timedelta(days=1))
start_time = st.sidebar.time_input("Start Time", datetime.now().time())
end_time = st.sidebar.time_input("End Time", (datetime.now() + timedelta(hours=1)).time())
details = st.sidebar.text_area("Medication details", "Take Crocin in the morning")


if st.sidebar.button("Set Reminder"):
    start_datetime = datetime.combine(date, start_time).strftime("%Y%m%dT%H%M%S")
    end_datetime = datetime.combine(date, end_time).strftime("%Y%m%dT%H%M%S")

    calendar_link = generate_google_calendar_link(title, start_datetime, end_datetime, details)

    st.sidebar.markdown(f"[📅 Add to Google Calendar]({calendar_link})", unsafe_allow_html=True)
# --- Apply Full Dark Mode Styling ---
if theme == "🌙 Dark Mode":
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f172a !important;  /* Dark background */
        color: #f1f5f9 !important;  /* Light text */
    }

    .stSidebar, .css-1d391kg, .css-1v0mbdj {
        background-color: #1e293b !important;  /* Darker sidebar */
        color: #f1f5f9 !important;  /* Light text for sidebar */
    }

    .stMarkdown, .markdown-text-container, .stTextInput>div>input, .stSelectbox {
        color: #f1f5f9 !important;  /* Light text for markdown, inputs, selects */
        background-color: #334155 !important;  /* Optional: slight background for inputs */
    }

    .stButton>button {
        background-color: #3B82F6 !important;  /* Blue button */
        color: white !important;
        border-radius: 8px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #60A5FA !important;  /* Lighter blue on hover */
        color: white !important;
    }

    h1, h2, h3, h4 {
        color: #93C5FD !important; /* Soft blue for headings */
    }
    </style>
""", unsafe_allow_html=True)



elif theme == "🌞 Light Mode":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: white !important;
            color: #1E3A8A !important;  /* Blue text */
            
        }

        .stSidebar, .css-1d391kg, .css-1v0mbdj {
            background-color: #f0f2f6 !important;
            color: #1E3A8A !important;  /* Blue text for sidebar */
        }

        .stMarkdown, .markdown-text-container, .stTextInput>div>input, .stSelectbox {
            color: #1E3A8A !important;  /* Blue for general text, input */
        }

        .stButton>button {
            background-color: #1E3A8A !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #3B82F6 !important;
            color: white !important;
        }

        h1, h2, h3, h4 {
            color: #2563EB !important; /* Deep blue for headings */
        }
        </style>
    """, unsafe_allow_html=True)

#Title of the app
st.title("🩺 DocBot - I'm here to help!")

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text if text else "No text found in the PDF."

def ask_mistral(content):
    try:
        response = mistral_client.chat.complete(
            model="mistral-large-latest", messages=[{"role": "user", "content": content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"



def generate_medical_response(user_prompt):
    try:
        #CSV loading
        medical_df = pd.read_csv('merckal.csv', 
                                names=['question', 'answer', 'source', 'focus_area'],
                                header=0)

        def clean_text(text):
            text = str(text).lower().strip()
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
            for word in ['what', 'are', 'the', 'how', 'to', 'do', 'does', 'can']:
                text = text.replace(word, '')
            return text.strip()

        user_prompt_clean = clean_text(user_prompt)
        medical_df['clean_question'] = medical_df['question'].apply(clean_text)


        matches = difflib.get_close_matches(user_prompt_clean, 
                                            medical_df['clean_question'], 
                                            n=1, cutoff=0.75)

        if matches:
            best_match = matches[0]
            matched_row = medical_df[medical_df['clean_question'] == best_match].iloc[0]

            question = str(matched_row['question']).strip()
            answer = str(matched_row['answer']).strip()
            source = str(matched_row['source']).strip() if not pd.isna(matched_row['source']) else "Not specified"
            focus_area = str(matched_row['focus_area']).strip() if not pd.isna(matched_row['focus_area']) else "General"

            # Response Limit: 200 words
            words = answer.split()
            if len(words) > 200:
                answer = " ".join(words[:200]) + "..."


            structured_info = (
                f"🔎 **Matched Question:**\n> {question}\n\n"
                f"💬 **Response from the data:**\n{textwrap.fill(answer, width=80)}\n\n"
                f"📚 **Source:** {source}\n"
                f"🧠 **Focus Area:** {focus_area}\n"
            )


            gemini_prompt = (
                f"The user asked: {user_prompt}\n\n"
                f"Here is some information retrieved from a medical book:\n"
                f"Q: {question}\nA: {answer}\n\n"
                f"Can you provide a deeper or more patient-friendly explanation, or additional advice?"
            )
            gemini_response = gemini_model.generate_content([gemini_prompt])


            return (
                f"📖 *Based on the book data, here's what I found:*\n\n"
                f"{structured_info}\n"
                f"🤖 *Here's what DocBot AI suggests additionally:*\n\n{gemini_response.text}"
            )
        
        else:
            gemini_response = gemini_model.generate_content([user_prompt])
            return f"🤖 *No exact match found from the book. Here's what DocBot AI suggests:*\n\n{gemini_response.text}"

    except Exception as e:
        return f"⚠️ Error: {e}"

def generate_title(content):
    try:
        response = gemini_model.generate_content(f"Summarize this conversation in a short title: {content}")
        return response.text.strip()
    except Exception as e:
        return "Untitled Query"

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError:
            return "API unavailable."

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def check_insurance_coverage(treatment, insurance_provider):
    prompt = f"""You are an AI assistant specializing in health insurance. A patient with {insurance_provider} insurance is asking if their treatment for {treatment} is covered. Provide a helpful response addressing:
    1. Likelihood of coverage
    2. Potential out-of-pocket costs
    3. Steps to verify coverage with their insurance provider
    4. Any general advice about insurance coverage for this treatment
    Be empathetic and informative in your response."""
    
    response = ask_mistral(prompt)
    return response

#bot
if mode == "Medical Chatbot":
#History
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar="🩺" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

#Query
    col1, col2 = st.columns([8, 2])
    with col1:
        user_input = st.chat_input("Ask a medical question...")
    with col2:
        uploaded_file = st.file_uploader("", type=["pdf", "jpg", "png", "txt"], label_visibility="collapsed")

    if uploaded_file:
        file_type = uploaded_file.type
        extracted_text = ""

        if "image" in file_type:
            image = Image.open(uploaded_file)
            extracted_text = extract_text_from_image(image)
        elif "pdf" in file_type:
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif "text" in file_type:
            extracted_text = uploaded_file.read().decode("utf-8")

        if extracted_text:
            ai_response = ask_mistral(extracted_text)
            st.session_state["messages"].append({"role": "user", "content": f"📄 Uploaded File Content:\n{extracted_text}"})
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Analyzing..."):
                    st.markdown(ai_response)
            st.session_state["messages"].append({"role": "assistant", "content": ai_response})

    elif user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing..."):
                response = generate_medical_response(user_input)
                st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

#Voice Input
    audio_value = st.audio_input("Record a voice message...")
    if audio_value is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_value.getbuffer())
        transcribed_text = speech_to_text("temp_audio.wav")
        if transcribed_text:
            # Clean the transcribed text
            cleaned_text = re.sub(r'\*+', '', transcribed_text).strip()
            st.session_state["messages"].append({"role": "user", "content": cleaned_text})
            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("Analyzing..."):
                    response = generate_medical_response(cleaned_text)
                    # Clean the response if needed
                    cleaned_response = re.sub(r'\*+', '', response).strip()
                    tts = gTTS(cleaned_response, lang="en")
                    tts.save("response.mp3")
                    with open("response.mp3", "rb") as f:
                        audio_bytes = f.read()
                    autoplay_audio(audio_bytes)
                    st.audio(audio_bytes, format="audio/mp3")
            st.session_state["messages"].append({"role": "assistant", "content": cleaned_response})

    if st.session_state["messages"] and not st.session_state.get("restoring_history", False):
        title = generate_title(st.session_state["messages"][-1]["content"])
        st.session_state["history"][title] = st.session_state["messages"].copy()
        st.session_state["messages"] = []
    st.session_state["restoring_history"] = False  # Reset flag after use

#Insurance

def estimate_coverage(treatment, provider):
    prompt = f"Does {provider} insurance cover {treatment}? Provide an AI-powered estimation with deductible insights."
    return ask_mistral(prompt)

def compare_policies(policy_1, policy_2):
    prompt = f"Compare these two insurance policies: {policy_1} vs {policy_2}. Give a concise comparison."
    return ask_mistral(prompt)

def calculate_deductible_cost(deductible, copay, treatment_cost):
    out_of_pocket = min(treatment_cost, deductible) + (treatment_cost * (copay / 100))
    return f"Estimated out-of-pocket cost: ${out_of_pocket:.2f}"

def pre_authorization_steps(treatment, provider):
    prompt = f"What are the steps for pre-authorization for {treatment} under {provider}?"
    return ask_mistral(prompt)

# --- INSURANCE COVERAGE CHECK MODE ---
if mode == "Insurance Coverage Check":
    st.header("Insurance Coverage Assistance")
    option = st.selectbox("Choose an insurance service", [
        "Estimate Coverage", "Compare Policies",
        "Deductible & Copay Calculator", "Pre-Authorization Steps"])
    
    
    
    if option == "Estimate Coverage":
        treatment = st.text_input("Treatment Inquiry:")
        provider = st.text_input("Insurance Provider:")
        if st.button("Estimate Coverage") and treatment and provider:
            response = estimate_coverage(treatment, provider)
            st.write(response)
    
    elif option == "Compare Policies":
        policy_1 = st.text_area("Policy 1 Details:")
        policy_2 = st.text_area("Policy 2 Details:")
        if st.button("Compare") and policy_1 and policy_2:
            response = compare_policies(policy_1, policy_2)
            st.write(response)
    
    elif option == "Deductible & Copay Calculator":
        deductible = st.number_input("Annual Deductible ($):", min_value=0)
        copay = st.number_input("Copay Percentage (%):", min_value=0, max_value=100)
        treatment_cost = st.number_input("Treatment Cost ($):", min_value=0)
        if st.button("Calculate Cost"):
            result = calculate_deductible_cost(deductible, copay, treatment_cost)
            st.write(result)
    
    elif option == "Pre-Authorization Steps":
        treatment = st.text_input("Treatment Inquiry:")
        provider = st.text_input("Insurance Provider:")
        if st.button("Get Steps") and treatment and provider:
            response = pre_authorization_steps(treatment, provider)
            st.write(response)
