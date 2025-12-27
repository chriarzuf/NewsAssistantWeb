import streamlit as st
import sys
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from newsapi import NewsApiClient
from newspaper import Article
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="AI News Assistant", layout="wide")

# --- SETUP NLTK (Cached) ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
    return True

setup_nltk()

# --- CARICAMENTO MODELLI (Cached) ---
@st.cache_resource
def load_models():
    """Carica i modelli una sola volta per non rallentare l'app"""
    print("Caricamento modelli AI in corso...")
    # Sentiment Analysis
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Summarization
    summarizer_model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("Modelli caricati!")
    return sentiment_model, summarizer_model

sentiment_analyzer, summarizer = load_models()

# --- FUNZIONI DI UTILITÀ ---

def preprocess_text(text):
    """Pulizia del testo per Word Cloud e Keywords"""
    if not text: return []
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    custom_stops = {'news', 'report', 'says', 'new', 'update', 'live', 'said', 'us', 'could', 'chars'}
    stop_words.update(custom_stops)
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
    return filtered

def get_sentiment_label(text):
    """Analizza sentiment di un testo breve"""
    try:
        # Tronchiamo a 512 caratteri per sicurezza
        return sentiment_analyzer(text[:512])[0]
    except:
        return None

def analyze_article_content(url):
    """Scarica e analizza l'articolo completo"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        article = Article(url)
        article.html = response.text
        article.download_state = 2
        article.parse()
        
        full_text = article.text
        if not full_text or len(full_text) < 100:
            return None

        # Summarization
        input_len = len(full_text.split())
        max_len = min(130, input_len // 2)
        summary = summarizer(full_text, max_length=max_len, min_length=30, truncation=True)[0]['summary_text']
        
        # Sentiment Full Text
        sentiment = sentiment_analyzer(full_text[:512])[0]
        
        # Keywords
        clean_tokens = preprocess_text(full_text)
        keywords = Counter(clean_tokens).most_common(5)
        
        return {
            "title": article.title,
            "summary": summary,
            "sentiment": sentiment,
            "keywords": keywords,
            "full_text": full_text
        }
    except Exception as e:
        st.error(f"Errore nell'analisi: {e}")
        return None

# --- INTERFACCIA STREAMLIT ---

st.title("📰 AI News Assistant & Daily Briefing")
st.markdown("Generazione automatica di rassegna stampa, Word Cloud e analisi sentiment tramite Deep Learning.")

# Sidebar per controlli
with st.sidebar:
    st.header("Impostazioni")
    api_key = st.text_input("Inserisci NewsAPI Key", value="f2473c71c92945a0add2e545acdd5ee0", type="password")
    category = st.selectbox("Scegli Categoria", ['general', 'technology', 'business', 'science', 'health', 'entertainment'])
    
    if st.button("📡 Scarica Notizie"):
        if not api_key:
            st.error("Inserisci una API Key valida.")
        else:
            newsapi = NewsApiClient(api_key=api_key)
            with st.spinner('Scaricamento notizie e generazione Word Cloud...'):
                try:
                    response = newsapi.get_top_headlines(category=category, language='en', page_size=30)
                    articles = response.get('articles', [])
                    
                    if articles:
                        st.session_state['articles'] = articles
                        st.session_state['category'] = category
                        st.success(f"Trovati {len(articles)} articoli!")
                    else:
                        st.warning("Nessun articolo trovato.")
                except Exception as e:
                    st.error(f"Errore API: {e}")

# --- MAIN PAGE: LOGICA ---

if 'articles' in st.session_state and st.session_state['articles']:
    articles = st.session_state['articles']
    
    # 1. SEZIONE OVERVIEW (WordCloud e Global Sentiment)
    st.divider()
    st.subheader(f"📊 Panoramica del Giorno: {st.session_state['category'].upper()}")
    
    # Calcolo dati aggregati
    all_text = ""
    sentiments = []
    
    for art in articles:
        text_content = f"{art['title']} {art['description'] or ''}"
        all_text += text_content + " "
        res = get_sentiment_label(art['title'])
        if res: sentiments.append(res['label'])
    
    # Colonne per metriche e grafico
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Calcolo percentuali sentiment
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        if total > 0:
            pos_pct = (sentiment_counts['POSITIVE'] / total) * 100
            neg_pct = (sentiment_counts['NEGATIVE'] / total) * 100
            
            st.metric(label="Umore Positivo", value=f"{pos_pct:.1f}%")
            st.metric(label="Umore Negativo", value=f"{neg_pct:.1f}%")
        else:
            st.write("Dati insufficienti per il sentiment.")

    with col2:
        # Generazione Word Cloud
        clean_tokens = preprocess_text(all_text)
        if clean_tokens:
            wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(clean_tokens))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Testo insufficiente per la Word Cloud.")

    # 2. SEZIONE DETTAGLIO ARTICOLI
    st.divider()
    st.subheader("🔍 Analisi Approfondita Articolo")
    
    # Creiamo una lista di titoli per il menu a tendina
    article_options = [f"{i+1}. {a['title']}" for i, a in enumerate(articles)]
    selected_option = st.selectbox("Seleziona un articolo da analizzare:", article_options)
    
    # Troviamo l'indice selezionato
    selected_index = article_options.index(selected_option)
    selected_article_data = articles[selected_index]
    
    # Bottone per analizzare
    if st.button("Avvia Analisi AI"):
        with st.spinner("Lettura e analisi in corso (potrebbe richiedere qualche secondo)..."):
            result = analyze_article_content(selected_article_data['url'])
            
            if result:
                # Layout Risultati
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown("### 📝 Riassunto")
                    st.info(result['summary'])
                    
                    st.markdown("### 📄 Testo Completo (Espandibile)")
                    with st.expander("Leggi articolo originale"):
                        st.write(result['full_text'])
                
                with c2:
                    st.markdown("### 🧠 AI Insights")
                    
                    # Sentiment Box
                    s_label = result['sentiment']['label']
                    s_score = result['sentiment']['score']
                    if s_label == 'POSITIVE':
                        st.success(f"**POSITIVO**\n\nConfidenza: {s_score:.2f}")
                    else:
                        st.error(f"**NEGATIVO**\n\nConfidenza: {s_score:.2f}")
                    
                    # Keywords Box
                    st.markdown("### 🔑 Keywords")
                    for word, freq in result['keywords']:
                        st.code(f"{word}: {freq}")
                    
                    st.markdown(f"[Vai alla fonte originale]({selected_article_data['url']})")
            else:
                st.error("Impossibile scaricare il testo dell'articolo (possibile Paywall o blocco anti-bot).")

else:
    st.info("👈 Inserisci la tua API Key nella barra laterale e clicca su 'Scarica Notizie' per iniziare.")
