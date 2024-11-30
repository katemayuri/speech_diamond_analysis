#running code
#code to generate labelled dataset for sentiment, keywords and trends on transcripts generated and store in output folder
import os
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from rake_nltk import Rake
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from tqdm import tqdm
import torch
from collections import Counter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Hugging Face's sentiment analysis pipeline with CUDA if available
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Pretrained sentiment model (fine-tuned on SST-2 dataset)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load sentiment pipeline, making sure it's using CUDA (GPU) if available
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# Download NLTK data (for sentence tokenization)
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize RAKE for Keyword Extraction
rake = Rake()

# Function to load and process files from the transcript folder
def process_transcripts(folder_path, output_folder, sentence_group_size=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    labeled_data = []
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Assuming your files are in .txt format
            file_path = os.path.join(folder_path, filename)
            output_file = os.path.join(output_folder, f"labeled_{filename}.json")
            
            with open(file_path, 'r') as file:
                content = file.read()
                
                # Split the content into sentences
                sentences = sent_tokenize(content)
                
                # Process each sentence
                file_labeled_data = []
                sentence_groups = [sentences[i:i+sentence_group_size] for i in range(0, len(sentences), sentence_group_size)]
                
                for group in sentence_groups:
                    sentiment_label = get_sentiment(' '.join(group))  # Perform sentiment analysis on the whole group
                    keywords = extract_keywords(' '.join(group))  # Extract keywords for the whole group
                    trends = extract_trends_using_lda(group)  # Extract trends for the whole group
                    
                    # for sentence in group:
                    #     # Associate the same sentiment, keywords, and trends with each sentence in the group
                    file_labeled_data.append({
                        'sentence': group,
                        'sentiment': sentiment_label,
                        'keywords': keywords,
                        'trends': trends
                    })
                
                # Save the labeled data for the current file
                save_labeled_data(file_labeled_data, output_file)

# Function to get sentiment label (positive, negative, neutral)
def get_sentiment(text, max_length=512):
    # Tokenize the text to measure its token length
    tokenized_text = tokenizer(text, truncation=False, return_tensors="pt")
    num_tokens = tokenized_text.input_ids.shape[1]
    
    if num_tokens > max_length:
        # Split the text into smaller chunks
        chunks = []
        chunk_size = max_length - 10  # Allow space for special tokens like [CLS] and [SEP]
        for i in range(0, num_tokens, chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)
        
        # Analyze each chunk separately and aggregate results
        sentiments = []
        for chunk in chunks:
            result = sentiment_analyzer(chunk)[0]
            sentiments.append(result['label'])
        
        # Determine final sentiment based on majority voting
        final_sentiment = max(set(sentiments), key=sentiments.count)
    else:
        # If the text length is within limits, analyze directly
        result = sentiment_analyzer(text)[0]
        final_sentiment = result['label']
    
    # Map the label to a human-readable sentiment
    if final_sentiment == 'POSITIVE':
        return 'positive'
    elif final_sentiment == 'NEGATIVE':
        return 'negative'
    else:
        return 'neutral'

# Function to extract keywords using RAKE and get top 5 keywords using RAKE Score
def extract_keywords(sentence, top_n=7):
    # Extract keywords with scores
    rake.extract_keywords_from_text(sentence)
    keyword_scores = rake.get_ranked_phrases_with_scores()
    
    # Sort keywords by score (descending order) and select top N
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[0], reverse=True)
    top_keywords = [keyword for score, keyword in sorted_keywords[:top_n]]
    
    return top_keywords

# Function to preprocess and clean the text
def preprocess_text(text, min_length=3):
    # Tokenize the sentence and remove stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # If the sentence after filtering has fewer than min_length tokens, skip it
    if len(filtered_tokens) < min_length:
        return ""
    
    return ' '.join(filtered_tokens)


# Function to train the LDA model and extract trends
def extract_trends_using_lda(sentences, num_topics=5, top_n = 7):
    # Preprocess all sentences
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Filter out empty sentences after preprocessing
    processed_sentences = [sentence for sentence in processed_sentences if sentence.strip()]
    
    if len(processed_sentences) == 0:
        return []  # If no valid sentences remain, return an empty list of trends

    # Convert the sentences into a Term-Document matrix using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_sentences)

    # Train the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the top words for each topic with their relevance scores
    feature_names = vectorizer.get_feature_names_out()
    topic_scores = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]  # Top 10 words for this topic
        top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
        topic_scores.extend(top_words)
    
    # Aggregate trends based on their scores
    word_scores = Counter()
    for word, score in topic_scores:
        word_scores[word] += score
    
    # Sort by relevance and return the top N trends
    sorted_trends = [word for word, score in word_scores.most_common(top_n)]
    return sorted_trends


# Save labeled dataset as JSON or CSV
def save_labeled_data(labeled_data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(labeled_data, json_file, indent=4)
        
    print(f"Labeled data saved to {output_file}")

# Example usage
folder_path = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Rapaport Podcasts/Transcripts"
output_folder = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Rapaport Podcasts/label_data"

# Process the transcripts and generate labeled data
process_transcripts(folder_path, output_folder)

print("Process complete!")
