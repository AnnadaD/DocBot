<p align="center"> <img width="235" alt="Image" src="https://github.com/user-attachments/assets/269a92fb-e93a-49d5-9a56-7a019345cf96" /></p> <h1 align="center">DocBot 🩺</h1> <p align="center"> DocBot is an intelligent medical assistant that combines AI-driven chat capabilities with document analysis, voice recognition, medication reminders, and insurance check features. With multi-modal input (PDFs, images, voice), personalized advice, and calendar integration, DocBot is your all-in-one healthcare AI. </p>



# 🧠 DocBot – Your AI-Powered Medical Assistant

DocBot is your intelligent medical companion that helps you understand your health better, check insurance coverage, set medication reminders, and interact with documents and images—all in one place.

## ✨ Features

### 🔍 AI-Powered Medical Assistant
- Ask any medical question and receive reliable, context-aware responses powered by Google's Vertex AI (Gemini) and Mistral
- Built using a curated medical dataset for trustworthy insights
- **New!** Multilingual support for non-English speakers

### 📄 Document & Image Analysis
- Upload PDFs or medical documents and extract text using `pdfplumber`
- Use OCR (via `pytesseract`) to extract text from images like prescriptions or reports
- Supports documents in multiple languages

### 🎙️ Voice to Text Interaction
- Speak your questions and get responses through a voice-enabled interface
- Uses `SpeechRecognition` and `gTTS` for voice input and output
- Supports multiple languages for voice interaction
- Perfect for hands-free interaction and accessibility

### 🗓️ Google Calendar Medication Reminders
- Schedule your medications with title, date, time, and details
- Generates a Google Calendar link so you can save your reminder instantly

### 💬 Dynamic Chat Modes
Switch between:
- Medical Chatbot
- Insurance Coverage Checker
- All from the sidebar with a seamless, dynamic interface

### 🌓 Custom Theme Support
- Toggle between light and dark themes
- Fully customizable UI via embedded CSS

## ⚙️ Technology Stack

| Technology | Description |
|------------|-------------|
| Streamlit | Frontend and UI framework |
| Vertex AI (Gemini) | Medical Q&A generation |
| Mistral AI | Additional contextual chatbot support |
| pytesseract | OCR for image-based text extraction |
| pdfplumber | PDF parser for medical reports |
| SpeechRecognition + gTTS | Voice input and text-to-speech output |
| Google Calendar API | Medication reminder integration |
| Pandas | Dataset querying and intelligent matching |

## Here's how it looks like

<img width="1440" alt="Image" src="https://github.com/user-attachments/assets/9bb75c7d-f867-4279-b0ab-f6fba87e5892" />

## 🧪 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/docbot.git
cd docbot
