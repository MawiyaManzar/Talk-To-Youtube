Talk to YouTube
Talk to YouTube is a Streamlit-based web application that allows users to interact with YouTube video transcripts conversationally. By pasting a YouTube video link, users can ask questions about the video's content, and the app provides answers based on the transcript using a retrieval-augmented generation (RAG) pipeline powered by LangChain and Google Gemini.

Features
Video Transcript Processing: Extracts transcripts from YouTube videos using youtube_transcript_api.
Conversational Interface: Ask questions about the video in a chat-like UI and get answers based on the transcript.
Local Embeddings: Uses Hugging Face's sentence-transformers for embedding generation, avoiding API quota limits.
Efficient Storage: Caches transcript embeddings in a FAISS vector store for fast retrieval.
User-Friendly: Intuitive Streamlit interface with clear feedback and error handling.
Free Tier Compatible: Works within Google Gemini's free tier limits for text generation.
Prerequisites
Python 3.8 or higher
A Google Cloud API key for the Gemini model (free tier)
A computer with internet access and sufficient memory for local embedding generation
Installation
Clone the Repository:
bash

Copy
git clone https://github.com/your-username/talk-to-youtube.git
cd talk-to-youtube
Create a Virtual Environment (optional but recommended):
bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash

Copy
pip install -r requirements.txt
The requirements.txt should include:
text

Copy
streamlit
langchain
langchain-community
langchain-google-genai
youtube-transcript-api
sentence-transformers
faiss-cpu
python-dotenv
tenacity
Set Up Environment Variables:
Create a .env file in the project root.
Add your Google API key for Gemini:
text

Copy
GOOGLE_API_KEY=your_google_api_key_here
Obtain a free API key from the Google Cloud Console.
Usage
Run the Application:
bash

Copy
streamlit run app.py
This opens the app in your default web browser (typically at http://localhost:8501).
Interact with the App:
Paste a YouTube video link (e.g., https://youtu.be/dQw4w9WgXcQ).
Wait for the transcript to load (you’ll see a success message).
Type a question in the chat input (e.g., “Summarize the video” or “What was the main topic?”).
View the response in the chat interface.
Ask follow-up questions to continue the conversation.
Clear Chat:
Click the “Clear Chat” button to reset the conversation history.
Example
Video URL: https://youtu.be/dQw4w9WgXcQ
Question: “What is the video about?”
Response: “The video is Rick Astley’s ‘Never Gonna Give You Up,’ a classic 1980s pop song. It’s famous for the ‘Rickrolling’ internet meme.”
Troubleshooting
Transcript Error: If you see “Could not retrieve transcript,” the video may not have English captions. Try another video (e.g., popular educational content).
Google API Error: If Gemini hits a quota limit, wait for the daily reset or reduce question frequency. Check usage in the Google Cloud Console.
Slow Performance: Ensure your computer has enough memory for local embeddings. If slow, reduce chunk_size in app.py (e.g., from 500 to 300).
Invalid URL: Ensure the YouTube link includes a valid video ID (11 characters, e.g., uaTfTikNe9w).
Limitations
Requires videos with English transcripts (auto-generated or manual).
Limited by Google Gemini’s free tier quota for text generation (typically 60–100 requests per minute).
Local embeddings may be slower on low-end hardware compared to cloud APIs.
No support for real-time video analysis or non-English transcripts in the current version.
Future Improvements
Add support for manual transcript uploads.
Implement voice input/output for a true “talk to” experience.
Support multiple languages by expanding transcript language options.
Add video metadata (title, description) to enhance context.
Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Built with LangChain, Streamlit, and Hugging Face.
Powered by Google Gemini and YouTube Transcript API.
Inspired by the need for accessible, conversational video analysis.
