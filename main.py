from dotenv import load_dotenv
load_dotenv()
import os
import requests
import arxiv
import json
import google.generativeai as genai
import streamlit as st
from typing import List, Dict, TypedDict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean
from pathlib import Path
from langgraph.graph import StateGraph, END

# --- Configuration ---
# The GOOGLE_API_KEY is now managed via Streamlit's secrets management.
# Set it in your .streamlit/secrets.toml file.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error("CRITICAL: GOOGLE_API_KEY is not set in Streamlit secrets.")
except (FileNotFoundError, KeyError):
    st.error("CRITICAL: .streamlit/secrets.toml not found or GOOGLE_API_KEY is missing.")
    GOOGLE_API_KEY = None


# --- Core Data Structures ---
@dataclass
class Paper:
    id: str
    source: str
    title: Optional[str] = None
    full_text: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None

class AgentState(TypedDict):
    user_query: str
    topic_list: List[str]
    papers_to_process: List[Dict[str, str]]
    processed_papers: Dict[str, Paper]
    classified_papers: Dict[str, List[str]]
    individual_summaries: Dict[str, str]
    topic_synthesis: Dict[str, str]
    podcast_scripts: Dict[str, str]
    audio_outputs: Dict[str, str]

# --- Agent Definitions ---
# Added print() statements for terminal logging alongside st.* calls.

class SearchAgent:
    def search(self, query: str, sort_by: str = "Relevance", max_results: int = 3) -> List[Dict[str, str]]:
        print(f"Searching ArXiv for '{query}' (Sort by: {sort_by})...")
        st.info(f"Searching ArXiv for '{query}' (Sort by: {sort_by})...")
        
        sort_criterion = arxiv.SortCriterion.Relevance if sort_by == "Relevance" else arxiv.SortCriterion.LastUpdatedDate
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_criterion)
            results = client.results(search)
            papers_to_process = [{"url": result.entry_id} for result in results]
            print(f"Found {len(papers_to_process)} papers.")
            st.success(f"Found {len(papers_to_process)} papers.")
            return papers_to_process
        except Exception as e:
            print(f"An error during ArXiv search: {e}")
            st.error(f"An error during ArXiv search: {e}")
            return []

class PaperProcessor:
    def _resolve_doi(self, doi: str) -> Optional[str]:
        """Resolves a DOI to a URL using the Crossref API."""
        print(f"Resolving DOI: {doi}...")
        st.info(f"Resolving DOI: {doi}...")
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Try to get the URL, prioritizing the PDF link if available
            pdf_url = data.get("message", {}).get("URL")
            if pdf_url:
                return pdf_url
            return data.get("message", {}).get("link", [{}])[0].get("URL")
        except Exception as e:
            print(f"Could not resolve DOI {doi}: {e}")
            st.error(f"Could not resolve DOI {doi}: {e}")
            return None

    def process_source(self, source_info: Dict[str, str]) -> Optional[Paper]:
        source_type, source_value = list(source_info.items())[0]
        print(f"Processing {source_type}: {source_value}...")
        st.info(f"Processing {source_type}: {source_value}...")
        
        full_text = None
        
        try:
            if source_type == "url":
                elements = partition(url=source_value, content_type="text/html")
                full_text = "\n\n".join([clean(el.text) for el in elements])
            elif source_type == "doi":
                resolved_url = self._resolve_doi(source_value)
                if resolved_url:
                    elements = partition(url=resolved_url, content_type="text/html")
                    full_text = "\n\n".join([clean(el.text) for el in elements])
            elif source_type == "pdf":
                # For PDF, the source_value is the path to the temp file
                elements = partition(filename=source_value)
                full_text = "\n\n".join([clean(el.text) for el in elements])

            if not full_text:
                print(f"Warning: Failed to extract text from {source_value}")
                st.warning(f"Failed to extract text from {source_value}")
                return None
            
            paper = Paper(id=str(source_value), source=str(source_value), full_text=full_text)
            return paper
            
        except Exception as e:
            print(f"Error processing source {source_value}: {e}")
            st.error(f"Error processing source {source_value}: {e}")
            return None

class SummarizerAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def summarize_and_enhance_paper(self, paper: Paper) -> Optional[str]:
        if not paper.full_text: return None
        print(f"Summarizing paper: {paper.id}...")
        st.info(f"Summarizing paper: {paper.id}...")
        prompt = f"Analyze the following paper text and provide a JSON object with keys: 'title', 'authors', 'abstract', and 'summary'. The summary should be a Markdown string with sections for Core Objective, Methodology, Key Findings, and a Citation.\n\nPaper Text (first 30k chars):\n---\n{paper.full_text[:30000]}"
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            parsed_data = json.loads(response.text)
            paper.title = parsed_data.get("title")
            paper.authors = parsed_data.get("authors", [])
            paper.abstract = parsed_data.get("abstract")
            summary = parsed_data.get("summary")
            print(f"Successfully summarized: {paper.title}")
            st.success(f"Successfully summarized: {paper.title}")
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            st.error(f"Error during summarization: {e}")
            return None

class TopicClassifierAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def classify_paper(self, paper: Paper, topic_list: List[str]) -> Optional[str]:
        if not paper.title or not paper.abstract: return None
        print(f"Classifying paper: {paper.title}...")
        st.info(f"Classifying paper: {paper.title}...")
        prompt = f"Classify this paper into ONE of these topics: {', '.join(topic_list)}. Respond with only the topic name.\n\nTitle: {paper.title}\nAbstract: {paper.abstract}\n\nTopic:"
        try:
            response = self.model.generate_content(prompt)
            topic = response.text.strip()
            return topic if topic in topic_list else None
        except Exception as e:
            print(f"Error during classification: {e}")
            st.error(f"Error during classification: {e}")
            return None

class SynthesizerAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def synthesize_topic(self, topic_name: str, summaries: List[str]) -> Optional[str]:
        if not summaries: return None
        print(f"Synthesizing report for topic: {topic_name}...")
        st.info(f"Synthesizing report for topic: {topic_name}...")
        summaries_text = "\n\n---\n\n".join(summaries)
        prompt = f"You are a research analyst. Synthesize these paper summaries on '{topic_name}' into a cohesive report. Identify themes, methodologies, and contrasting views. Cite findings using the citations from each summary.\n\nSummaries:\n{summaries_text}\n\nSynthesized Report:"
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"An error occurred during synthesis: {e}")
            st.error(f"An error occurred during synthesis: {e}")
            return None

class AudioGenerationAgent:
    def __init__(self, llm_model_name="gemini-1.5-flash"):
        self.llm_model = genai.GenerativeModel(llm_model_name)

    def create_podcast_script(self, topic_name: str, report_text: str) -> Optional[str]:
        print(f"Creating podcast script for topic: {topic_name}...")
        st.info(f"Creating podcast script for topic: {topic_name}...")
        prompt = f"You are a podcast host. Transform this report on '{topic_name}' into an engaging, conversational script with an intro, main body, and outro. Use cues like `[intro music]`.\n\nReport:\n{report_text}\n\nPodcast Script:"
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error during script generation: {e}")
            st.error(f"Error during script generation: {e}")
            return None

    def generate_audio(self, script_text: str, output_filename: str) -> Optional[str]:
        print(f"Generating audio with free TTS service: {output_filename}...")
        st.info(f"Generating audio with free TTS service: {output_filename}...")
        url = "https://marytt.reverso.net/read_text"
        payload = {"language": "en-US", "voice": "pavoque-styles", "text": script_text, "output_type": "audio/mpeg"}
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.post(url, data=payload, headers=headers, stream=True)
            response.raise_for_status()
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully saved audio to {output_filename}")
            st.success(f"Successfully saved audio to {output_filename}")
            return output_filename
        except Exception as e:
            print(f"Error during free TTS API call: {e}")
            st.error(f"Error during free TTS API call: {e}")
            return None

# --- Graph Setup ---
@st.cache_resource
def build_graph():
    if not GOOGLE_API_KEY: return None
    agents = {
        'search': SearchAgent(), 'processor': PaperProcessor(), 'summarizer': SummarizerAgent(),
        'classifier': TopicClassifierAgent(), 'synthesizer': SynthesizerAgent(), 'audio': AudioGenerationAgent()
    }
    def search_node(state):
        print("\n--- 🔎 SEARCH ---")
        # The user_query now contains the filter option
        query, sort_by = state['user_query']
        return {"papers_to_process": agents['search'].search(query, sort_by=sort_by)}
    
    def process_and_summarize_node(state):
        print("\n--- 📄 PROCESS & SUMMARIZE ---")
        processed_papers, summaries, classified = {}, {}, defaultdict(list)
        for source in state['papers_to_process']:
            paper = agents['processor'].process_source(source)
            if paper and (summary := agents['summarizer'].summarize_and_enhance_paper(paper)):
                if topic := agents['classifier'].classify_paper(paper, state['topic_list']):
                    processed_papers[paper.id], summaries[paper.id] = paper, summary
                    classified[topic].append(paper.id)
        return {"processed_papers": processed_papers, "individual_summaries": summaries, "classified_papers": dict(classified)}
    
    def synthesize_node(state):
        print("\n--- 🧬 SYNTHESIZE ---")
        synthesis = {}
        for topic, pids in state['classified_papers'].items():
            summaries = [state['individual_summaries'][pid] for pid in pids if pid in state['individual_summaries']]
            if summaries and (report := agents['synthesizer'].synthesize_topic(topic, summaries)):
                synthesis[topic] = report
                # Save the report to a file in the output directory
                report_filename = f"output/{topic.replace(' ', '_')}_report.md"
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Saved synthesis report to {report_filename}")
                st.info(f"Saved synthesis report to {report_filename}")
        return {"topic_synthesis": synthesis}
    
    def audio_generation_node(state):
        print("\n--- 🎙️ GENERATE AUDIO ---")
        scripts, outputs = {}, {}
        for topic, report in state['topic_synthesis'].items():
            if script := agents['audio'].create_podcast_script(topic, report):
                scripts[topic] = script
                audio_filename = f"output/{topic.replace(' ', '_')}_podcast.mp3"
                if audio_path := agents['audio'].generate_audio(script, audio_filename):
                    outputs[topic] = audio_path
        return {"podcast_scripts": scripts, "audio_outputs": outputs}

    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("process_and_summarize", process_and_summarize_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("generate_audio", audio_generation_node)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "process_and_summarize")
    workflow.add_edge("process_and_summarize", "synthesize")
    workflow.add_edge("synthesize", "generate_audio")
    workflow.add_edge("generate_audio", END)
    return workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Research Agent System", layout="wide")
st.title("📚 Research Paper Summarization Agent System")

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Get the graph application
app = build_graph()

if not app:
    st.warning("Application could not be started. Please check your GOOGLE_API_KEY in .streamlit/secrets.toml")
else:
    st.sidebar.header("Controls")
    
    input_method = st.sidebar.radio("Input Method", ["Search ArXiv", "Process URL/DOI", "Upload PDF"])
    
    papers_to_process_input = []
    run_button_pressed = False

    if input_method == "Search ArXiv":
        query = st.sidebar.text_input("Research Query", "AI in drug discovery")
        sort_by = st.sidebar.radio("Sort by", ["Relevance", "Recency"])
        run_button_pressed = st.sidebar.button("Run Research Agent")
        if run_button_pressed:
            # The search node will handle this
            pass

    elif input_method == "Process URL/DOI":
        url_or_doi = st.sidebar.text_input("Enter a URL or DOI", "10.1038/s41586-021-03583-4")
        run_button_pressed = st.sidebar.button("Process Paper")
        if run_button_pressed:
            if url_or_doi.startswith("10."):
                 papers_to_process_input = [{"doi": url_or_doi}]
            else:
                 papers_to_process_input = [{"url": url_or_doi}]

    elif input_method == "Upload PDF":
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
        run_button_pressed = st.sidebar.button("Process PDF")
        if run_button_pressed and uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open(os.path.join("output", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            papers_to_process_input = [{"pdf": os.path.join("output", uploaded_file.name)}]

    topics_str = st.sidebar.text_input("Topics (comma-separated)", "Computational Biology, AI in Healthcare, Pharmaceuticals")
    
    if run_button_pressed:
        if not topics_str:
            st.sidebar.error("Please provide topics.")
        else:
            topics_list = [topic.strip() for topic in topics_str.split(',')]
            
            if input_method == "Search ArXiv":
                inputs = {"user_query": (query, sort_by), "topic_list": topics_list}
            else:
                # For direct processing, we bypass the search node
                # This requires a more complex graph, so for now we'll just process it directly
                # and then run the rest of the graph.
                # A more robust solution would use conditional edges in the graph.
                inputs = {"papers_to_process": papers_to_process_input, "topic_list": topics_list}

            with st.spinner("Research agents are at work... This may take a few minutes."):
                final_state = None
                
                # Simplified flow for now: if not searching, we manually run the first step
                if input_method != "Search ArXiv":
                    # Manually run the processing and summarization
                    intermediate_state = process_and_summarize_node(inputs)
                    # Then run the rest of the graph
                    remaining_graph_inputs = {**inputs, **intermediate_state}
                    for s in app.stream(remaining_graph_inputs, {"recursion_limit": 100}, stream_mode="values"):
                         if "generate_audio" in s:
                            final_state = s["generate_audio"]
                else:
                    for s in app.stream(inputs):
                        final_state = list(s.values())[0]

            st.header("✅ Research Complete!")
            
            if final_state:
                st.subheader("Results")
                
                if final_state.get("topic_synthesis"):
                    for topic, report in final_state["topic_synthesis"].items():
                        st.markdown(f"---")
                        st.markdown(f"### Report for Topic: {topic}")
                        st.markdown(report)
                        
                        audio_path = final_state.get("audio_outputs", {}).get(topic)
                        if audio_path and os.path.exists(audio_path):
                            st.audio(audio_path)
                        else:
                            st.warning(f"Audio file for '{topic}' not found.")
                else:
                    st.warning("No synthesis reports were generated.")
            else:
                st.error("The agent run failed to produce a final state.")
