from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Handles the data pre-processing before the analysis
# Even if this class has generic functions that can be used generally, this class is applied in only in topic analysis


class MyPreProcessor:

    # Constructor - Initiates the the appropriate instances regarding the stop-words, stemming and lemmatization
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        # self.stop_words.extend(
        #     ["from", "subject", "re", "edu", "use", "not", "would", "say", "could", "_", "be", "know", "good", "go",
        #      "get", "do", "done", "try", "many", "some", "nice", "thank", "think", "see", "rather", "easy", "easily",
        #      "lot", "lack", "make", "want", "seem", "run", "need", "even", "right", "line", "even", "also", "may",
        #      "take", "come"])
        self.porter_stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    # Performs tokenization and stop-word cleaning - Private function
    def data_pre_processing_nltk(self, text):
        word_tokens = word_tokenize(text)
        word_tokens = [w.lower() for w in word_tokens if w.isalpha()]
        text = [w for w in word_tokens if not w in self.stop_words]
        return text

    # Performs tokenization and stop-word cleaning, stemming and lemmatization - Private function
    def data_pre_processing_nltk_extended(self, text):
        word_tokens = word_tokenize(text)
        word_tokens = [w.lower() for w in word_tokens if w.isalpha()]
        word_tokens = [w for w in word_tokens if not w in self.stop_words]
        word_tokens = [self.porter_stemmer.stem(w) for w in word_tokens]
        # text = [self.lemmatizer.lemmatize(w) for w in word_tokens]
        return word_tokens

