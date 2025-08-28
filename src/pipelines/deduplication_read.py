import argparse
import itertools
import requests
import re, string
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer # pip install sentence-transformers
from bs4 import BeautifulSoup
from similarius import extract_text_ressource as og_extract_text_ressource, sk_similarity # pip install similarius


# Connects to website
def get_website(website: str):
    """Fetch website, fallback to http:// if no scheme given"""
    if not website.startswith(("http://", "https://")):
        website = "http://" + website
    try:
        return requests.get(website, timeout=5)
    except requests.exceptions.RequestException:
        return None

# Extracts only necessary text from website i.e excludes menus, ads, banners etc.    
def clean_text_from_html(body: str) -> str:
    """Extract visible text from HTML, keeping only meaningful tags"""
    soup = BeautifulSoup(body, "lxml")

    # Drop irrelevant tags
    # Still up for debate which tags to exclude considering some sites might employ unconventional structures, change as needed
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form", "input"]):
        tag.decompose()

    texts = []
    # For now, only consider certain tags as meaningful content
    for element in soup.find_all(["p", "article", "main", "h1", "h2", "h3", "li"]):
        txt = element.get_text(" ", strip=True)
        if txt and len(txt.split()) > 5:
            texts.append(txt)

    return " ".join(texts)


# Cleans up extracted text
def normalize_text(text: str, max_words: int = 500) -> str:
    """Lowercase, strip punctuation/numbers, truncate to max_words"""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text

# Overrides similarius function to include improved text cleaning
def extract_text_ressource(html: str):
    """Wrapper around similarius.extract_text_ressource with improved text cleaning"""
    raw_text, resources = og_extract_text_ressource(html)
    clean_text = normalize_text(clean_text_from_html(html))
    return clean_text, resources

# Adds in argument for custom input
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--file",
    default=r"dataset\test.txt",  
    help="Path to a text file where each line contains a URL to a website"
)
args = parser.parse_args()

# NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reads in websites from file
with open(args.file, "r", encoding="utf-8") as f:
    websites = [line.strip() for line in f if line.strip()]

# Loops through websites in file and attempts to extract text
site_texts = {}
for site in websites:
    resp = get_website(site)
    if not resp:
        print(f"[-] {site} is unreachable, skipping...")
        continue
    text, _ = extract_text_ressource(resp.text)
    if text.strip():
        site_texts[site] = text
    else:
        print(f"[-] {site} has no extractable text")

# Embedding for semantic similarity
embeddings = {
    site: model.encode([content])[0]
    for site, content in site_texts.items()
}

test = "I liked the movie."
test_vec = model.encode([test])[0]

# Output for loop for similarity scores
for site_a, site_b in itertools.combinations(embeddings.keys(), 2):
    sim_score_sem = distance.cosine(embeddings[site_a], embeddings[site_b])
    sim_score_lex = sk_similarity(site_texts[site_a], site_texts[site_b])
    print(f"\n********** {site_a} <-> {site_b} **********")
    print("\n")
    print(f"Lexical Similarity (TF-IDF, sk_similarity) = {sim_score_lex}%")
    print(f"Semantic Similarity (SentenceTransformer) = {sim_score_sem:.4f}")
    print("\n")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("\n")
    