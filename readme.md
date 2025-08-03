Research Paper Summarization Multi-Agent System
This project is a multi-agent system designed to find, analyze, summarize, and create audio podcasts from research papers. It uses a combination of a Streamlit UI, the LangGraph framework for agent orchestration, and Google's Gemini models for language processing tasks.

System Architecture
The application is built around a multi-agent system orchestrated by LangGraph. A central AgentState object is passed through a graph where each node represents a specialized agent performing a specific task. This stateful approach allows for a clear and robust data flow from initial search to final output.

Agent Roles
Search Agent: Queries the ArXiv API to find relevant research papers based on a user's query and filtering options (Relevance or Recency).

Paper Processor Agent: Ingests papers from various sources (URLs, DOIs, or uploaded PDFs). It uses the unstructured.io library to extract clean, readable text from these sources.

Summarizer Agent: Uses a Gemini model to read the extracted text, identify the paper's title, authors, and abstract, and generate a structured summary.

Topic Classifier Agent: Takes the enhanced paper data and classifies it into a user-defined topic list based on its title and abstract.

Synthesizer Agent: Groups all papers by their assigned topic and uses a Gemini model to create a single, cohesive report that discusses themes, trends, and contrasting findings.

Audio Generation Agent:

First, it uses a Gemini model to convert the formal synthesized report into a conversational, engaging podcast script.

Then, it uses a free, public Text-to-Speech (TTS) service to convert the script into an MP3 audio file.

Technology Stack
UI Framework: Streamlit

Agent Orchestration: LangGraph

Language Models: Google Gemini 1.5 Flash

Paper Search: ArXiv API

Text Extraction: Unstructured.io

Containerization: Docker

Setup and Execution
You can run this application in two simple commands using Docker, which is the recommended method.

Prerequisites
Docker: You must have Docker installed and running on your machine.

Google AI API Key: You need a valid API key from Google AI Studio.

Step 1: Create the Secrets File
In the root of your project directory, create a new folder named .streamlit.

Inside the .streamlit folder, create a file named secrets.toml.

Add your Google AI API key to this file as follows:

GOOGLE_API_KEY = "your_api_key_here"

Step 2: Build and Run with Docker
Open your terminal in the project's root directory and run the following commands:

Build the Docker image: This command packages your application and all its dependencies into an image named research-agent.

docker build -t research-agent .

Run the Docker container: This command starts the application from the image you just built and makes it accessible on your local machine.

docker run -p 8501:8501 research-agent

Step 3: Access the Application
Once the container is running, open your web browser and navigate to:

http://localhost:8501

How to Use the Application
The Streamlit interface provides several ways to process research papers:

Search ArXiv: The default method. Enter a query (e.g., "quantum computing"), choose a sorting filter, provide your topics, and click "Run Research Agent".

Process URL/DOI: Select this option to process a single paper from a direct web link or a Digital Object Identifier (DOI).

Upload PDF: Select this option to upload a PDF file of a research paper directly from your computer.

The application will display real-time status updates in the UI and detailed logs in your terminal. Once complete, it will show the synthesized reports and provide an audio player for each generated podcast. All generated reports (.md) and audio files (.mp3) are also saved to an output folder in your project directory.

Limitations and Future Improvements
Single Search Source: The system currently only searches ArXiv. It could be expanded to include other sources like Semantic Scholar, PubMed, or Google Scholar.

Free TTS Service: The audio generation relies on a public, free TTS service which may have rate limits or be less reliable than a dedicated API. This could be upgraded to a more robust commercial service.

Basic Error Handling: While the system is functional, error handling could be made more sophisticated to handle edge cases, such as failed downloads or malformed papers, more gracefully.