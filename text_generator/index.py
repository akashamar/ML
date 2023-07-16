import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()

# Step 1: Read the contents of the text file
with open('content.txt', 'r') as file:
    text = file.read()

# Step 2: Preprocess the text
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def preprocess_text(text):
    # Tokenize into sentences
    sentences = sent_tokenizer.tokenize(text)

    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    preprocessed_sentences = [sentence.translate(translator).lower() for sentence in sentences]

    return preprocessed_sentences

preprocessed_text = preprocess_text(text)

# Step 3: Map user questions to relevant information
def get_relevant_info(question, preprocessed_text):
    # Preprocess the question
    preprocessed_question = preprocess_text(question)[0]  # Take the first sentence of the question

    # Create TF-IDF vectorizer and compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_question] + preprocessed_text)

    # Compute cosine similarity between question and text sentences
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    # Find the most similar sentence in the text
    most_similar_sentence_idx = similarity_scores.argmax()
    most_similar_sentence = preprocessed_text[most_similar_sentence_idx]

    return most_similar_sentence

# Step 4: Create API endpoint
@app.get('/generate_response')
def generate_response(user_question: str):
    relevant_info = get_relevant_info(user_question, preprocessed_text)

    generated_text = relevant_info

    return {'response': generated_text}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
