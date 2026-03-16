import re
from collections import Counter
import spacy

nlp = spacy.load("uk_core_news_sm")


def filter(path):
    """
    Зчитує текст з файлу та видаляє службові слова
    (наприклад, 'ТСН', 'Тема') на початку рядків.
    Повертає список очищених рядків.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        split = l.split()
        for _ in range(6):
            if split and (split[0] == 'ТСН' or split[0] == "Тема"):
                split.pop(0)
            if split:
                split.pop(0)
        lines[i] = " ".join(split)
    return lines


def normalize(lines):
    """
    Приводить текст до нижнього регістру
    та видаляє розділові знаки.
    Повертає список нормалізованих рядків.
    """
    normalized_lines = []
    for l in lines:
        normalized = l.lower()
        normalized = re.sub(r"[.,!?;:\"'(){}\[\]<>—–]", '', normalized)
        normalized_lines.append(normalized.strip())
    return normalized_lines


# ===============================
# ТОКЕНІЗАЦІЯ
# ===============================

def tokenize_sentences(lines):
    """
    Виконує токенізацію тексту по реченнях.
    Повертає список речень.
    """
    sentences = []
    for line in lines:
        doc = nlp(line)
        sentences.extend([sent.text for sent in doc.sents])
    return sentences


def tokenize_words(sentences):
    """
    Виконує токенізацію по словах.
    Видаляє пунктуацію, пробіли та числа.
    Повертає список списків слів.
    """
    tokenized = []
    for sentence in sentences:
        doc = nlp(sentence)
        words = [
            token.text
            for token in doc
            if not token.is_punct
            and not token.is_space
            and not token.like_num
        ]
        tokenized.append(words)
    return tokenized


def tokenize_characters(sentences):
    """
    Виконує токенізацію по символах.
    Повертає список списків символів.
    """
    char_tokens = []
    for sentence in sentences:
        char_tokens.append(list(sentence))
    return char_tokens


# ===============================
# ВИДАЛЕННЯ СТОП-СЛІВ
# ===============================

def remove_stopwords(tokenized_words):
    """
    Видаляє стоп-слова з токенізованого тексту.
    Повертає список списків слів без стоп-слів.
    """
    cleaned = []
    for sentence in tokenized_words:
        doc = nlp(" ".join(sentence))
        no_stops = [token.text for token in doc if not token.is_stop]
        cleaned.append(no_stops)
    return cleaned


# ===============================
# ЛЕМАТИЗАЦІЯ
# ===============================

def lemmatize(tokenized_words):
    """
    Виконує лематизацію слів.
    Повертає список списків лем.
    """
    lemmatized = []
    for sentence in tokenized_words:
        doc = nlp(" ".join(sentence))
        lemmas = [token.lemma_ for token in doc]
        lemmatized.append(lemmas)
    return lemmatized


# ===============================
# ТОП СЛІВ
# ===============================

def top_words(tokenized_text, n=10):
    """
    Обчислює n найчастіших слів у тексті.
    Повертає список кортежів (слово, кількість).
    """
    all_words = [word for sentence in tokenized_text for word in sentence]
    counter = Counter(all_words)
    return counter.most_common(n)


def save_to_file(data, filename):
    """
    Зберігає список або текстові дані у файл.
    Якщо передано список списків — об'єднує елементи через пробіл.
    """
    with open(filename, "w", encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    f.write(" ".join(item) + "\n")
                else:
                    f.write(str(item) + "\n")
        else:
            f.write(str(data))


# ===============================
# СТЕМІНГ
# ===============================

def simple_stem(word):
    """
    Проста реалізація стемінгу для української мови.
    Видаляє типові закінчення.
    """
    endings = [
        "ами", "ями", "ах", "ях",
        "ів", "ев",
        "ий", "ій",
        "ого", "ому",
        "и", "і", "а", "у", "ю", "е", "о"
    ]

    for ending in sorted(endings, key=len, reverse=True):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]

    return word


def stem_text(tokenized_text):
    """
    Виконує стемінг для списку списків слів.
    Повертає список списків основ слів.
    """
    stemmed = []
    for sentence in tokenized_text:
        stemmed_sentence = [simple_stem(word) for word in sentence]
        stemmed.append(stemmed_sentence)
    return stemmed


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    filtered = filter("result.txt")
    save_to_file(filtered, "filtered.txt")
    normalized = normalize(filtered)
    save_to_file(normalized, "normalized.txt")
    sentences = tokenize_sentences(normalized)

    words = tokenize_words(sentences)
    save_to_file(words, "words.txt")

    chars = tokenize_characters(sentences)

    no_stops = remove_stopwords(words)
    save_to_file(no_stops, "no_stopwords.txt")

    lemmas = lemmatize(no_stops)
    save_to_file(lemmas, "lemmatized.txt")

    stemmed = stem_text(lemmas)
    save_to_file(stemmed, "6_stemmed.txt")

    top10 = top_words(stemmed)
    with open("top10.txt", "w", encoding="utf-8") as f:
        for word, count in top10:
            f.write(f"{word}: {count}\n")

    print("Файли успішно збережені")
    print("\nПо реченнях\n" + str(sentences[:5]))
    print("\nПо словах\n" + str(words[:5]))
    print("\nПо символах\n" + str(chars[:5]))