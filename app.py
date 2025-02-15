import streamlit as st
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re


corpus = ""
# Initialize session state for the model
if 'model' not in st.session_state:
    st.session_state.model = None

# Custom CSS to style the button
st.markdown("""
    <style>
    .stButton>button {
        width: 200px;
        border: 2px solid #ff6347;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def word_2_vec(corpus):
    # Remove punctuation and other non-important characters
    corpus = re.sub(r'[^\w\s]', '', corpus)
    my_text = sent_tokenize(corpus)
    tokenized_sentences = [[word for word in word_tokenize(sentence) if word.lower() not in stopwords.words('english')]
    for sentence in my_text]


    #To use Skip-gram, set the 'sg' parameter to True
    model = Word2Vec(tokenized_sentences,window=5,min_count=1,workers=4,max_vocab_size=100,sg=False)
    st.session_state.model = model
   
            
    

st.title("Word2Vec(CBOW & Skip-gram)")



# Allow only .txt files to be uploaded
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

# Display the content of the uploaded file
if uploaded_file is not None:
    # Read the file as a string
    corpus = uploaded_file.read().decode("utf-8")
    st.text_area("File Content", corpus, height=100)
else:
    st.write("Please upload a .txt file.")


# Create a button
if st.button("Create Model"):
    # Call the function when the button is clicked
    word_2_vec(corpus)
   


first_word = st.text_input('Enter first word')
second_word = st.text_input('Enter second word')

if first_word and second_word:
    similarity = st.session_state.model.wv.similarity(first_word,second_word)
    st.write(similarity)