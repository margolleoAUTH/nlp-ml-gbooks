from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Handles the data pre-processing before the analysis
# Even if this class has generic functions that can be used generally, this class is applied in only in topic analysis


class MyPreProcessor:

    # Constructor - Initiates the the appropriate instances regarding the stop-words, stemming and lemmatization
    def __init__(self):
        stop_words = stopwords.words("english")
        stop_words.extend([
            "from", "subject", "re", "edu", "use", "not", "would", "say", "could", "_", "be", "know", "good", "go",
            "get", "do", "done", "try", "many", "some", "nice", "thank", "think", "see", "rather", "easy", "easily",
            "lot", "lack", "make", "want", "seem", "run", "need", "even", "right", "line", "even", "also", "may",
            "take", "come",
            "so", "his", "t", "y", "ours", "herself",
            "your", "all", "some", "they", "i", "of", "didn",
            "them", "when", "will", "that", "its", "because",
            "while", "those", "my", "don", "again", "her", "if",
            "further", "now", "does", "against", "won", "same",
            "a", "during", "who", "here", "have", "in", "being",
            "it", "other", "once", "itself", "hers", "after", "re",
            "just", "their", "himself", "theirs", "whom", "then", "d",
            "out", "m", "mustn", "where", "below", "about", "isn",
            "shouldn", "wouldn", "these", "me", "to", "doesn", "into",
            "the", "until", "she", "am", "under", "how", "yourself",
            "couldn", "ma", "up", "than", "from", "themselves", "yourselves",
            "off", "above", "yours", "having", "mightn", "needn", "on",
            "too", "there", "an", "and", "down", "ourselves", "each",
            "hadn", "ain", "such", "ve", "did", "be", "or", "aren", "he",
            "should", "for", "both", "doing", "this", "through", "do", "had",
            "own", "but", "were", "over", "not", "are", "few", "by",
            "been", "most", "no", "as", "was", "what", "s", "is", "you",
            "shan", "between", "wasn", "has", "more", "him", "nor",
            "can", "why", "any", "at", "myself", "very", "with", "we",
            "which", "hasn", "weren", "haven", "our", "ll", "only",
            "o", "before"
            "a",  "able",  "about",  "above",  "according",  "accordingly",  "across",  "actually",
            "after",  "afterwards",  "again",  "against",  "ain't",  "all",  "allow",  "allows",
            "almost", "alone", "along", "already", "also", "although", "always",  "am", "among",
            "amongst", "an", "and", "another", "any", "anybody",  "anyhow", "anyone", "anything",
            "anyway", "anyways", "anywhere", "apart",  "appear", "appreciate", "appropriate", "are",
            "aren't", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "available",
            "away", "awfully", "be", "became", "because", "become", "becomes", "becoming", "been", "before",
            "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better",
            "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant", "can't",
            "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com", "come",
            "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
            "contains", "corresponding", "could", "couldn't", "course", "c's", "currently", "definitely",
            "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing", "don",
            "done", "don't", "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else",
            "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every",
            "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except",
            "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former",
            "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting",
            "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadn't",
            "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
            "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "here's", "hereupon",
            "hers", "herself", "he's", "hi", "him", "himself", "his", "hither", "hopefully", "how",
            "howbeit", "however", "how's", "i", "i'd", "ie", "if", "ignored", "i'll", "i'm", "immediate",
            "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
            "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself",
            "i've", "just", "keep", "keeps", "kept", "know", "known", "knows", "last", "lately", "later",
            "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
            "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe", "me", "mean",
            "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must",
            "mustn't", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need",
            "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non",
            "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously",
            "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto",
            "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside",
            "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please",
            "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather",
            "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
            "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second",
            "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
            "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "since", "six", "so", "some", "somebody",
            "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",
            "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t",
            "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats",
            "that's", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
            "thereby", "therefore", "therein", "theres", "there's", "thereupon", "these", "they", "they'd",
            "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those",
            "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took",
            "toward", "towards", "tried", "tries", "truly", "try", "trying", "t's", "twice", "two", "un",
            "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use",
            "used", "useful", "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs",
            "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome", "well", "we'll", "went", "were",
            "we're", "weren't", "we've", "what", "whatever", "what's", "when", "whence", "whenever",
            "when's", "where", "whereafter", "whereas", "whereby", "wherein", "where's", "whereupon",
            "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "who's",
            "whose", "why", "why's", "will", "willing", "wish", "with", "within", "without", "wonder",
            "won't", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours",
            "yourself", "yourselves", "you've", "zero"
             ])
        self.stop_words = set(stop_words)
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
        # word_tokens = [self.porter_stemmer.stem(w) for w in word_tokens]
        text = [self.lemmatizer.lemmatize(w) for w in word_tokens]
        return text

