CSCI 6518 Deep Speech Project: Diamond Market Analysis
Overview
This project focuses on analyzing sentiments and trends in the diamond market by leveraging advanced Natural Language Processing (NLP) techniques. Using podcasts and videos sourced from the Rapaport Community, we explore how recent socio-economic factors influence diamond prices, sales, and trading.

Objectives
Extract and analyze market sentiments from audio-visual content.
Identify and generate market trends using NLP models.
Compare the performance of fine-tuned models against baseline models for both sentiment and trend analysis.
Data Collection
The source data includes videos and podcasts from the Rapaport Community Website, a trusted network offering ethical and transparent insights into diamond and jewelry markets. The content spans the past six months, focusing on socio-economic factors affecting the diamond industry.

Methodology
1. Audio Transcription
Tool Used: OpenAI's Whisper (medium) model.
Each audio file was transcribed into human-readable text, forming the basis for further analysis.
2. Labeled Dataset Creation
We prepared a labeled dataset for fine-tuning by:

Sentiment Analysis: Using BERT to classify sentiments in the transcriptions.
Keyword Extraction: Leveraging RAKE (Rapid Automatic Keyword Extraction) to identify relevant keywords.
Trend Analysis: Training an LDA (Latent Dirichlet Allocation) model to extract the top 7 trends from grouped sentences.
3. Model Fine-Tuning
Two tasks were addressed using fine-tuned models:

Sentiment Analysis: Fine-tuned DistilBERT and compared results with RoBERTa (baseline).
Trend Extraction: Fine-tuned KEYBART and compared results with Facebook BART (baseline).
Results
The fine-tuned DistilBERT model demonstrated significant improvements over the baseline RoBERTa for sentiment analysis.
Similarly, KEYBART outperformed Facebook BART for trend extraction in terms of coherence and relevance.