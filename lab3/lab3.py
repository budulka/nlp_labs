import re
import random
import spacy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# КОНСТАНТИ


PATH_EN = "/home/budulka/Documents/nlp/lab3/faust_en.txt"
PATH_UA = "/home/budulka/Documents/nlp/lab3/faust_ukr.txt"
BASE    = "/home/budulka/Documents/nlp/lab3"

EN_PHRASE = "Stay then, thou art so beautiful"
UA_PHRASE = "Зупинись, мить, ти прекрасна"

POS_FILTERS = {
    "ALL":       None,
    "NOUN":      {"NOUN"},
    "NOUN+VERB": {"NOUN", "VERB"},
}

ANCHOR_WORDS = {
    "EN": ["soul", "evil", "love"],
    "UA": ["душа", "зло",  "любов"],
}

COLORS = ['#aa0000', '#bb00bb', '#00cccc', '#dddd00', '#00ee00']


# ЧИТАННЯ ТА ОЧИСТКА


def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines, text


def clean_text(text, lang='en'):
    text = text.lower()
    if lang == 'en':
        text = re.sub(r'[^a-z\s]', ' ', text)
    else:
        text = re.sub(r'[^а-яіїєґ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_lines(lines, lang='en'):
    cleaned = [clean_text(line, lang) for line in lines]
    return [line for line in cleaned if len(line.split()) > 1]


# ЛЕМАТИЗАЦІЯ ТА POS


def is_valid(lemma, lang):
    pat = r'^[a-z]{2,}$' if lang == 'en' else r'^[а-яіїєґ]{2,}$'
    return bool(re.match(pat, lemma))


def parse_lines(lines, nlp, lang):
    result = []
    for doc in nlp.pipe(lines, batch_size=256):
        tokens = []
        for t in doc:
            if not t.is_stop and is_valid(t.lemma_.lower(), lang):
                tokens.append((t.lemma_.lower(), t.pos_))
        if tokens:
            result.append(tokens)
    return result


def filter_by_pos(parsed, allowed_pos=None):
    result = []
    for sent in parsed:
        tokens = []
        for lemma, pos in sent:
            if allowed_pos is None or pos in allowed_pos:
                tokens.append(lemma)
        if tokens:
            result.append(tokens)
    return result

# СЕГМЕНТИ


def make_lemma_text(sents, start_frac=0, end_frac=1.0):
    n     = len(sents)
    start = int(n * start_frac)
    end   = int(n * end_frac)
    chunk = sents[start:end]

    words = []
    for sent in chunk:
        for token in sent:
            words.append(token)
    return " ".join(words)


def lemmatize_phrase(text, nlp, lang, allowed_pos=None):
    doc    = nlp(clean_text(text, lang))
    lemmas = []
    for t in doc:
        if t.is_stop:
            continue
        if not is_valid(t.lemma_.lower(), lang):
            continue
        if allowed_pos is not None and t.pos_ not in allowed_pos:
            continue
        lemmas.append(t.lemma_.lower())
    return " ".join(lemmas)


def build_segments(lang, filter_name, lemmas_by_pos, nlp_en, nlp_ua):
    sents       = lemmas_by_pos[lang][filter_name]
    nlp         = nlp_en if lang == "EN" else nlp_ua
    l           = 'en'   if lang == "EN" else 'ua'
    phrase_raw  = EN_PHRASE if lang == "EN" else UA_PHRASE
    allowed_pos = POS_FILTERS[filter_name]
    return {
        "Частина 1":    make_lemma_text(sents, 0,   0.5),
        "Частина 2":    make_lemma_text(sents, 0.5, 1.0),
        "Повний текст": make_lemma_text(sents, 0,   1.0),
        "Фраза":        lemmatize_phrase(phrase_raw, nlp, l, allowed_pos),
    }


# TF-IDF


def print_sim_matrix(labels, matrix, title):
    print(f"\n  {title}:")
    print(f"  {'':16}", end='')
    for label in labels:
        print(f"{label:>14}", end='')
    print('\n')
    for i, label in enumerate(labels):
        print(f"  {label:16}", end='')
        for j in range(len(labels)):
            print(f"{matrix[i][j]:>14.3f}", end='')
        print()


def tfidf(lemmas_by_pos, nlp_en, nlp_ua):
    print("\n=== TF-IDF: КОСИНУСНА СХОЖІСТЬ СЕГМЕНТІВ ===")

    for lang in ["EN", "UA"]:
        for filter_name in POS_FILTERS:
            segs   = build_segments(lang, filter_name, lemmas_by_pos, nlp_en, nlp_ua)
            labels = list(segs.keys())
            corpus = list(segs.values())
            if not all(corpus):
                print(f"\n  [{lang}/{filter_name}] фраза порожня після фільтру — пропускаємо")
                continue
            sim = cosine_similarity(TfidfVectorizer().fit_transform(corpus))
            print_sim_matrix(labels, sim, f"TF-IDF [{lang} / {filter_name}]")

    print("\n  EN vs UA в спільному TF-IDF просторі")
    en_all    = build_segments("EN", "ALL", lemmas_by_pos, nlp_en, nlp_ua)
    ua_all    = build_segments("UA", "ALL", lemmas_by_pos, nlp_en, nlp_ua)
    combined  = list(en_all.values()) + list(ua_all.values())
    cross_sim = cosine_similarity(TfidfVectorizer().fit_transform(combined))

    print(f"\n  {'':14}", end='')
    for label in ua_all:
        print(f"{'UA:' + label:>18}", end='')
    print()
    for i, label in enumerate(en_all):
        print(f"  {'EN:' + label:14}", end='')
        for j in range(4, 8):
            print(f"{cross_sim[i][j]:>18.3f}", end='')
        print()


# WORD2VEC

def segment_vector(text, model):
    vecs = [model.wv[w] for w in text.split() if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


def word2vec(lemmas_by_pos, nlp_en, nlp_ua):
    print("\n=== WORD2VEC: КОСИНУСНА СХОЖІСТЬ СЕГМЕНТІВ ===")

    for lang in ["EN", "UA"]:
        for filter_name in POS_FILTERS:
            sents  = lemmas_by_pos[lang][filter_name]
            model  = Word2Vec(sentences=sents, vector_size=100, window=5, min_count=2, sg=0, epochs=50)
            segs   = build_segments(lang, filter_name, lemmas_by_pos, nlp_en, nlp_ua)
            labels = list(segs.keys())
            vecs   = np.array([segment_vector(t, model) for t in segs.values()])
            sim    = cosine_similarity(vecs)
            print_sim_matrix(labels, sim, f"Word2Vec [{lang} / {filter_name}]")


def most_similar(lemmas_by_pos):
    print("\nНайбільша схожість")
    for lang, anchor_list in ANCHOR_WORDS.items():
        sents = lemmas_by_pos[lang]["ALL"]
        model = Word2Vec(sentences=sents, vector_size=100, window=5, min_count=2, sg=0, epochs=50)
        print(f"\n  {lang}:")
        for word in anchor_list:
            similar = model.wv.most_similar(word, topn=5)
            print(f"\n    Найближчі до '{word}':")
            for w, score in similar:
                print(f"      {w:20} {score:.4f}")


def tfidf_top_words(lemmas_by_pos, nlp_en, nlp_ua):
    print("\n=== TF-IDF: ТОП СЛОВА ++=")

    for lang in ["EN", "UA"]:
        segs     = build_segments(lang, "ALL", lemmas_by_pos, nlp_en, nlp_ua)
        labels   = list(segs.keys())
        corpus   = list(segs.values())
        vec      = TfidfVectorizer()
        matrix   = vec.fit_transform(corpus)
        features = vec.get_feature_names_out()

        print(f"\n  {lang}:")
        for i, label in enumerate(labels):
            row     = matrix[i].toarray()[0]
            top_idx = row.argsort()[-3:][::-1]

            print(f"\n    {label}:")
            for j in top_idx:
                if row[j] > 0:
                    print(f"      {features[j]:20} {row[j]:.2f}")

# ВІЗУАЛІЗАЦІЯ

def plot_w2v(lang, parsed, save_path, seed=42):
    sents = filter_by_pos(parsed, None)
    model = Word2Vec(sentences=sents, vector_size=100, window=5, min_count=2, sg=0, epochs=50)

    seen      = set()
    adj_words = []
    for sent in parsed:
        for lemma, pos in sent:
            if pos == "ADJ" and lemma in model.wv and lemma not in seen:
                adj_words.append(lemma)
                seen.add(lemma)

    random.seed(seed)
    words  = random.sample(adj_words, min(30, len(adj_words)))
    vecs   = np.array([model.wv[w] for w in words])
    coords = PCA(n_components=2, random_state=seed).fit_transform(vecs)

    _, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Word2Vec — {lang}", fontsize=13, pad=12)

    for i, (word, (x, y)) in enumerate(zip(words, coords)):
        ax.scatter(x, y, color=COLORS[i % len(COLORS)], s=60, zorder=3, alpha=0.8)
        ax.annotate(word, (x, y), fontsize=7.5, ha='left', va='bottom',
                    xytext=(3, 2), textcoords='offset points')

    ax.axhline(0, color='#dddddd', linewidth=0.5)
    ax.axvline(0, color='#dddddd', linewidth=0.5)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Збережено: {save_path}")


# MAIN

if __name__ == "__main__":
    nlp_en = spacy.load("en_core_web_sm")
    nlp_ua = spacy.load("uk_core_news_sm")

    en_lines_raw, _ = read_txt(PATH_EN)
    ua_lines_raw, _ = read_txt(PATH_UA)
    en_lines = clean_lines(en_lines_raw, 'en')
    ua_lines = clean_lines(ua_lines_raw, 'ua')

    print("\nЛематизація")
    en_parsed = parse_lines(en_lines, nlp_en, 'en')
    ua_parsed = parse_lines(ua_lines, nlp_ua, 'ua')
    print(f"  EN: {sum(len(s) for s in en_parsed)} токенів | "
          f"UA: {sum(len(s) for s in ua_parsed)} токенів")

    print("\n=== POS TAGGING ===")
    for lang_label, parsed in [("EN", en_parsed), ("UA", ua_parsed)]:
        print(f"\n{lang_label}:")
        print(f"  {'Токен':22} {'POS'}")
        count = 0
        for sent in parsed[:5]:
            for lemma, pos in sent:
                print(f"  {lemma:22} {pos}")
                count += 1
                if count >= 20:
                    break
            if count >= 20:
                break

    lemmas_by_pos = {
        "EN": {f: filter_by_pos(en_parsed, p) for f, p in POS_FILTERS.items()},
        "UA": {f: filter_by_pos(ua_parsed, p) for f, p in POS_FILTERS.items()},
    }

    tfidf(lemmas_by_pos, nlp_en, nlp_ua)
    word2vec(lemmas_by_pos, nlp_en, nlp_ua)
    most_similar(lemmas_by_pos)
    tfidf_top_words(lemmas_by_pos, nlp_en, nlp_ua)

    print("\nPCA ВІЗУАЛІЗАЦІЯ W2V")
    plot_w2v("EN", en_parsed, f"{BASE}/pca_en.png")
    plot_w2v("UA", ua_parsed, f"{BASE}/pca_ua.png")

   
    # ТОНАЛЬНІСТЬ

    print("\n=== ТОНАЛЬНІСТЬ: TF-IDF + VADER ===\n")

    vader = SentimentIntensityAnalyzer()
    word_scores = vader.lexicon 

    full_text = " ".join(en_lines)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([full_text])
    features = vectorizer.get_feature_names_out()
    row = matrix[0].toarray()[0]

    score = 0.0
    for i, word in enumerate(features):
        if word in word_scores:
            score += row[i] * word_scores[word]

    label = "позитивне" if score > 0 else "негативне" if score < 0 else "нейтральне"
    print(f"  Весь текст:      {score:+.4f}  → {label}")

    adj_only = " ".join(lemma for sent in en_parsed for lemma, pos in sent if pos == "ADJ")
    matrix2 = vectorizer.transform([adj_only])
    row2 = matrix2[0].toarray()[0]
    score2 = sum(row2[i] * word_scores[word] for i, word in enumerate(features) if word in word_scores)
    label2 = "позитивне" if score2 > 0 else "негативне" if score2 < 0 else "нейтральне"
    print(f"  Лише прикметники: {score2:+.4f}  → {label2}")