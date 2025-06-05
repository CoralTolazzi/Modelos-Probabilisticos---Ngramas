import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def run():
    oraciones = obtener_corpus('corpus.txt')
    mostrar_corpus("Corpus", oraciones)

    oraciones_limpias = limpiar_corpus(oraciones)
    mostrar_corpus("Clean corpus", oraciones_limpias)

    freqs = n_grama(oraciones_limpias)
    graficar_comparacion_ngrams(freqs, top_n=10)


def obtener_corpus(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        lineas = [line.strip() for line in f if line.strip()]
    return lineas


def mostrar_corpus(txt, corpus):
    print(f"\n{txt}:")
    for s in corpus:
        print(s)


def limpiar_corpus(docs):
    oraciones_limpias = []
    for s in docs:
        oracion_limpia = " ".join(lematizar(sacar_stopwords_esp(word_tokenize(s))))
        if oracion_limpia:
            oraciones_limpias.append(oracion_limpia)
    return oraciones_limpias


def sacar_stopwords_esp(texto):
    esp = stopwords.words("spanish")
    texto_limpio = [w.lower() for w in texto if w.lower() not in esp
                  and w not in string.punctuation
                  and not w.isnumeric()
                  and w not in ["«", "»", "(", ")", "...", '—', '“', '”', '…', '.', ',', '¡', '¿', "),"]]
    return texto_limpio


def lematizar(texto):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]
    return lemmatized_text


def get_wordnet_pos(word):
    # Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def n_grama(oraciones_limpias):
    print(f"\nn-grama:")
    vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=2)
    X = vectorizer.fit_transform(oraciones_limpias)
    vocab = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    freqs = Counter(dict(zip(vocab, counts)))
    print(list(freqs.keys()))
    print("\ncantidad de N-Gramas ( bi-gramas y tri-gramas):",len(freqs))
    return freqs


def graficar_comparacion_ngrams(freqs, top_n=10):
    top = freqs.most_common(top_n)

    ngrams, values2 = zip(*top) if top else ([], [])

    np.arange(len(ngrams))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.barh(ngrams[::-1], values2[::-1], color='skyblue')
    plt.title("Top 2-gramas y 3-gramas")
    plt.xlabel("Frecuencia")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
