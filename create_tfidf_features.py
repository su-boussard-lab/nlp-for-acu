import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from negspacy.termsets import termset
from src.utils.config import config
from negspacy.negation import Negex

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

nlp1 = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_sci_sm")
ts = termset("en_clinical")
nlp2.add_pipe(
    "negex",
    config={
        "chunk_prefix": ["no"],
    },
    last=True,
)


def main(random_state: int = 42) -> None:
    """Main function which performs the TF-IDF feature extraction
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    # Load feature matrix and the train inidices
    feature_matrix = feature_matrix = (
        pd.read_csv(config.data.tabular_path, low_memory=False)
        .sort_values(by="PAT_DEID")
        .set_index("PAT_DEID")
        .drop("DEMO_INDEX_PRE_CHE", axis=1)
    )
    TRAIN_IDS = pd.read_csv(config.data.train_ids)["PAT_DEID"]

    # Read in the full notes (only BERT preprocessing applied to them)
    df_notes = pd.read_csv(config.data.notes_path).set_index("PAT_DEID")

    # Preprocess the notes
    """
    df_notes_anonymized = df_notes.note.map(lambda x: preprocess_I(x))
    df_notes_anonymized.to_csv("./data/df_notes_anonymized.csv")
    print("Preprocessing I (anonymization): done")
    df_notes_negated = df_notes_anonymized.map(lambda x: preprocess_II(x))
    df_notes_negated.to_csv("./data/df_notes_negated.csv")
    print("Preprocessing II (negation): done")
    """
    df_notes_negated = (
        pd.read_csv("./data/df_notes_negated.csv").set_index("PAT_DEID").note
    )
    df_notes_tokens = df_notes_negated.map(lambda x: preprocess_III(x))
    # df_notes_tokens.to_csv("./data/df_notes_tokens.csv")
    print("Preprocessing III (tokenization): done")

    # Find the most frequent occurences
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 1),
        stop_words=stopwords.words("english")
        + ["md", "cc", "dr", "stanford", "pt", "mg", "cm"],
        tokenizer=word_tokenize,
    )
    count_vectorizer.fit(df_notes_tokens.loc[df_notes.index.intersection(TRAIN_IDS)])
    X = count_vectorizer.transform(df_notes_tokens)
    count_vect_df = pd.DataFrame(
        X.todense(), columns=count_vectorizer.get_feature_names()
    )

    # Create a set with the N most frequent words
    N = config.tfidf.n_most_frequent
    N_most_freq_words = set(count_vect_df.sum().nlargest(N).index)

    # Weight the N most occuring terms with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        vocabulary=N_most_freq_words,
        sublinear_tf=True,
    )
    tfidf_vectorizer.fit(
        df_notes_tokens.loc[df_notes_tokens.index.intersection(TRAIN_IDS)]
    )
    tfidf_vect_df = pd.DataFrame(
        data=tfidf_vectorizer.transform(df_notes_tokens).todense(),
        columns=tfidf_vectorizer.get_feature_names(),
        index=df_notes_tokens.index,
    )

    # Combine with original feauture Matrix, containing the tabular data
    new_names = {}
    for c in tfidf_vect_df.columns:
        new_names[c] = "WORD_" + c
    tfidf_vect_df.rename(columns=new_names, inplace=True)

    combined_feature_matrix = feature_matrix.join(tfidf_vect_df, how="left")
    combined_feature_matrix.to_csv(config.data.data_path)


def negator(text: str):
    """Apply negation to medical terms found
    Args:
        text (str): the input string that needs to be processed
    Returns:
           string (str): string with entity linking
    """
    doc = nlp2(text)
    newString = text
    for e in reversed(doc.ents):
        if e._.negex:
            newString = (
                newString[: e.start_char]
                + f"NOT_{e}"
                + newString[e.start_char + len(e.text) :]
            )

    return newString


def abbreviation_transformer(input_string: str):
    """Apply abbreviation to medical terms found
    Args:
        input_string (str): the input string that needs to be processed
    Returns:
        abbrv_string (str): string with abbreviations
    """
    doc = nlp(input_string)
    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)
    return " ".join(altered_tok)


def filter_POS(
    input_string: str,
    pos_filters: list = ["ADV, AUX, ADP", "DET", "INTJ", "PRON"],
) -> str:
    """Used spacy to do POS tagging and remove pos types that are in the filter list
    Args:
        input_string (str): the input string that needs to be processed
        pos_filters (list): list of strings of POS types that need to be filered
    Returns:
        filtered_string (str)
    """
    doc = nlp1(input_string)
    filtered_string = ""
    for token in doc:
        if token.pos_ in pos_filters:
            new_token = ""  # f"<{token.pos_}>"
        elif token.pos_ == "PUNCT":
            new_token = token.text
        else:
            new_token = " {}".format(token.text)
        filtered_string += new_token
    filtered_string = filtered_string[1:]
    return filtered_string


def remove_entities(
    text: str,
    type_filters: list = ["PERSON", "ORG", "DATE", "CARDINAL", "QUANTITY", "TIME"],
):
    """Used spacy to remove certain entity types
    Args:
        input_string (str): the input string that needs to be processed
        filters (list): list of strings of POS types that need to be filered
    Returns:
        filtered_string (str)
    """
    doc = nlp1(text)
    newString = text
    for e in reversed(doc.ents):
        if e.label_ in type_filters:
            newString = (
                newString[: e.start_char]
                # + f"<{e.label_}>"
                + newString[e.start_char + len(e.text) :]
            )
    return newString


def preprocess_I(text: str, threshold: int = 1) -> str:
    """Preprocess text, such that the medical notes look somewhat okayish, and can also be fed to ClinicalBERT
       Requires nlp = spacy.load("en_core_web_sm")
    Args:
        text (str): input text, unprocessed
        threshold (int, 1): threshold for length of a word
    Returns:
        processed_text (str): processed text
    """
    # re.IGNORECASE ignoring cases
    # compilation step to escape the word for all cases
    compiled = re.compile(re.escape("stanford hospital and clinics"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford hospitals and clinics"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford cancer center"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford thoracic oncology"), re.IGNORECASE)
    text = compiled.sub("", text)

    # Replace weird character:
    text = text.replace("¿", "")

    # 2 spaces are indicative for a new line, someone gone wrong in saving
    text = text.replace("  ", "\n")

    # POS tagging and filtering
    type_filters = ["PERSON", "ORG", "DATE", "TIME"]
    text = remove_entities(text, type_filters=type_filters)

    return text


def preprocess_II(text: str, threshold: int = 1) -> str:
    """Preprocess text, with negation and special character removal
       NOTE: Should have gone through preproces_I before.
             Requires nlp = spacy.load("en_core_sci_sm")
                      nlp.add_pipe("negex")
    Args:
        text (str): input text, unprocessed
        threshold (int, 1): threshold for length of a word
    Returns:
        processed_text (str): processed text
    """
    text = text.replace("  ", "\n")
    text = negator(text)
    pos_filters = ["AUX", "ADP", "DET", "INTJ", "PRON"]
    text = filter_POS(text, pos_filters=pos_filters)
    return text


def preprocess_III(text: str, threshold: int = 1) -> str:
    """Simple preprocessing and standartization
    Args:
        text (str): input text, unprocessed
        threshold (int, 1): threshold for length of a word
    Returns:
        processed_text (str): processed text
    """
    # remove special characters
    text = re.sub("\\W", " ", text)

    # Make space consistent
    text = " ".join(text.split())

    # Lower case
    text = text.lower()

    # Remove digits
    # text = "".join([i for i in text if not i.isdigit()])

    # Remove single characters
    # text = " ".join([word for word in text.split(" ") if len(word) > threshold])

    # normalize certain words
    text = re.sub("\\s+(in|the|all|for|and|on)\\s+", " ", text)

    # Lemmatize words
    text = " ".join(
        [wordnet_lemmatizer.lemmatize(word=word) for word in re.split("\\s+", text)]
    )
    return text


if __name__ == "__main__":
    main(random_state=config.seed)
