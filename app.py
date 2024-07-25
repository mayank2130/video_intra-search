import re
import os
import requests
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Hugging Face Inference API setup
hf_token = ""
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_model(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?(?:\S+)"
    match = re.match(regex, url)
    if match:
        return url.split("v=")[-1][:11]
    return None

# Function to get video transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None

# Function to create context windows
def create_context_windows(transcript, window_size=10):
    context_windows = []
    for i in range(len(transcript)):
        start = max(0, i - window_size // 2)
        end = min(len(transcript), i + window_size // 2 + 1)
        window = transcript[start:end]
        context_windows.append({
            'start': window[0]['start'],
            'end': window[-1]['start'] + window[-1]['duration'],
            'text': ' '.join([segment['text'] for segment in window])
        })
    return context_windows

# Function to search transcription
def search_transcription(context_windows, query):
    vectorizer = TfidfVectorizer()
    context_vectors = vectorizer.fit_transform([window['text'] for window in context_windows])
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, context_vectors)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    results = [context_windows[i] for i in top_indices if similarities[i] > 0.1]

    return results

# Updated generate_response function
def generate_response(context, query):
    prompt = f"""AI assistant is a brand new, powerful, human-like artificial intelligence.
    The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
    AI is a well-behaved and well-mannered individual.
    AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user.
    AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
    START CONTEXT BLOCK
    {context}
    END OF CONTEXT BLOCK
    AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
    AI will focus more on giving answers in BULLET POINTS THAN paragraphs.
    AI WILL FOCUS MORE GIVING EXAMPLES FROM THE CONTEXT BLOCK.
    If the context does not provide the answer to question, the AI assistant will say, "I'm sorry, but I don't know the answer to that question".
    AI assistant will not apologize for previous responses, but instead will indicated new information was gained.
    AI assistant will not invent anything that is not drawn directly from the context.
    
    Human: {query}
    AI:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
        }
    }

    print("Sending request to Hugging Face API...")
    response = query_model(payload)
    print("Received response from Hugging Face API:", response)

    if isinstance(response, list) and 'generated_text' in response[0]:
        full_response = response[0]['generated_text']
        ai_message = full_response.split('AI:')[-1].split('Human:')[0].strip()
        return ai_message
    else:
        print("Unexpected response format:", response)
        return "Sorry, I couldn't generate a response. Please try again."
# Streamlit app
def main():
    st.title("Advanced YouTube Intra-Video Search")

    # Check for HF_TOKEN at the start of the app
    if not hf_token:
        st.error("Please set the HF_TOKEN environment variable with your Hugging Face token.")
        return

    url = st.text_input("Enter YouTube URL:")

    if url:
        video_id = extract_video_id(url)

        if video_id:
            try:
                yt = YouTube(url)
                video_title = yt.title
                st.write(f"Video Title: {video_title}")

                transcript = get_transcript(video_id)

                if transcript:
                    context_windows = create_context_windows(transcript)

                    query = st.text_input("Ask a question about the video:")

                    if query:
                        results = search_transcription(context_windows, query)

                        if results:
                            st.write("Here's what I found:")
                            for result in results:
                                start_minutes = int(result['start']) // 60
                                start_seconds = int(result['start']) % 60
                                print("\n\n\nSENDING THE TRANSCRIPT:-------------------------------------\n\n\n",result['start'])
                                response = generate_response(result['text'], query)
                                
                                st.write(f"Time: {start_minutes:02d}:{start_seconds:02d}")
                                st.write(f"Response: {response}")

                            # Display full video
                            st.video(url)

                            # Provide clickable timestamps
                            st.write("Jump to relevant parts:")
                            for result in results:
                                start_minutes = int(result['start']) // 60
                                start_seconds = int(result['start']) % 60
                                timestamp = f"{start_minutes:02d}:{start_seconds:02d}"
                                if st.button(timestamp):
                                    st.video(f"{url}&t={int(result['start'])}s")
                        else:
                            st.write("No relevant results found.")
                else:
                    st.write("Unable to fetch transcript. The video might not have captions available.")
            except Exception as e:
                st.write(f"An error occurred: {str(e)}")
        else:
            st.write("Invalid YouTube URL")

if __name__ == "__main__":
    main()