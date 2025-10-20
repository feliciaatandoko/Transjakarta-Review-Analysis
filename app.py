import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re, emoji
from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt
import colorsys
import time
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import altair as alt
from bertopic import BERTopic
import stanza
import io

# ===============================
# Konfigurasi halaman
# ===============================
st.set_page_config(
    page_title="🚍 Transjakarta Review Insight",
    layout="wide",
)

# --- Header utama ---
st.markdown("# 🚍 Transjakarta Review Insight")
st.markdown("Analyze user reviews to uncover sentiment trends and popular discussion topics about Transjakarta")
st.caption("Upload Data -> Analyze Sentiment -> Analyze Topic")

# ==============================
# Load model & tokenizer
# ==============================
@st.cache_resource
def load_sentiment_model():
    repo_id = "feliciaatandoko/model_indobert"
    sentiment_model = BertForSequenceClassification.from_pretrained(repo_id)
    sentiment_tokenizer = BertTokenizer.from_pretrained(repo_id)
    return sentiment_model, sentiment_tokenizer

sentiment_model, sentiment_tokenizer = load_sentiment_model()

@st.cache_resource
def load_topic_model_neg():
    model = BERTopic.load("model_topic_neg")
    return model

topic_model_neg = load_topic_model_neg()

@st.cache_resource
def load_topic_model_net():
    model = BERTopic.load("model_topic_net")
    return model

topic_model_net = load_topic_model_net()

@st.cache_resource
def load_topic_model_pos():
    model = BERTopic.load("model_topic_pos")
    return model

topic_model_pos = load_topic_model_pos()

# ==============================
# Text Cleaning
# ==============================
contraction_map = {
    "sy": "saya", "aq": "aku", "ak": "aku", "gue": "saya", "gw": "saya",
    "gua": "saya", "lu": "kamu", "lo": "kamu", "elu": "kamu", "km": "kamu",
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "udh": "sudah",
    "blm": "belum", "jg": "juga", "tp": "tapi", "trs": "terus", "krn": "karena",
    "klo": "kalau", "sm": "sama", "aj": "saja", "aja": "saja", "bgt": "banget",
    "gitu": "begitu", "kyk": "seperti", "tj": "transjakarta", "trnsjkt": "transjakarta",
    "tije": "transjakarta", "jawabbb": "jawab", "app": "aplikasi", "apps": "aplikasi", "apk": "aplikasi"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub("#\w+", " ", text)
    text = re.sub("@\w+", " ", text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = emoji.replace_emoji(text, replace=' ')
    text = re.sub(r'\b\w\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    expanded_words = [contraction_map.get(w, w) for w in words]
    return " ".join(expanded_words)

# ==============================
# Stopword Removal
# ==============================
factory = StopWordRemoverFactory()
stopwords_list = factory.get_stop_words()
stopword_remover = factory.create_stop_word_remover()
additional_stopwords = ["yang", "nya", "ya", "udah", "min", "apa"]
additional_stopwords_topic = ["yang", "nya", "ya", "udah", "min", "apa", "transjakarta"]

def remove_stopwords(text):
    tokens = text.split()
    tokens_clean = [word for word in tokens
                    if word not in stopwords_list and word not in additional_stopwords]
    return " ".join(tokens_clean)

def remove_stopwords_topic(text):
    tokens = text.split()
    tokens_clean = [word for word in tokens
                    if word not in stopwords_list and word not in additional_stopwords_topic]
    return " ".join(tokens_clean)

# ==============================
# Lemmatization
# ==============================
@st.cache_resource
def load_lemmatizer():
    stanza.download('id', processors='tokenize,pos,lemma')
    nlp = stanza.Pipeline(lang='id', processors='tokenize,pos,lemma')
    return nlp

def lemmatize_text(nlp, text):
    doc = nlp(text)
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            lemmas.append(word.lemma)
    return " ".join(lemmas)

# ==============================
# Prediction function - sentiment
# ==============================
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}

# ==============================
# Prediction function - topic negatif
# ==============================
def predict_topic_neg(text):
    topics, _ = topic_model_neg.transform([text])
    return topics[0]

label_map_topic_neg = {
    -1: "Outlier",
    0: "Layanan Transjakarta",
    1: "Sistem Pembayaran",
    2: "Waktu Tunggu",
    3: "Ketersediaan Armada",
    4: "Sistem Pengumuman",
    5: "Aplikasi Transjakarta",
    6: "Fasilitas Halte"
}

# ==============================
# Prediction function - topic netral
# ==============================
def predict_topic_net(text):
    topics, _ = topic_model_net.transform([text])
    return topics[0]

label_map_topic_net = {
    -1: "Outlier",
    0: "Panduan Rute",
    1: "Jadwal Operasional Bus",
    2: "Informasi Sistem Pembayaran"
}

# ==============================
# Prediction function - topic positif
# ==============================
def predict_topic_pos(text):
    topics, _ = topic_model_pos.transform([text])
    return topics[0]

label_map_topic_pos = {
    -1: "Outlier",
    0: "Kenyamanan Transportasi dan Supir",
    1: "Apresiasi Pelayanan Petugas",
    2: "Pengalaman Positif Layanan Transjakarta",
    3: "Ekspresi Pujian",
    4: "Ekspansi Rute dan Mobilitas"
}

# ==============================
# WordCloud generator
# ==============================
def generate_wordcloud(texts, colormap, max_words):
    text_combined = " ".join(texts)
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        colormap=colormap, 
        max_words=max_words
    ).generate(text_combined)
    
    return wc

# ==============================
# State management
# ==============================
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "sentiment_done" not in st.session_state:
    st.session_state.sentiment_done = False
if "topic_done" not in st.session_state:
    st.session_state.topic_done = False

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3 = st.tabs([
    "📁 Upload Data",
    "📊 Sentiment Analysis",
    "💡 Topic Analysis"
])


# ==============================
# Tab 1 - Upload Data
# ==============================
with tab1:
    st.subheader("📁 Upload Data")
    st.info("Upload a **.csv** file with **one text column** of user reviews")

    uploaded_file = st.file_uploader("Drag and drop your CSV file here", type=["csv"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] != 1:
            st.error("⚠️ CSV file should contain only one text column")
        else:
            col_name = df.columns[0]
            
            st.session_state.uploaded_data = df
            st.success(f"File `{uploaded_file.name}` uploaded successfully!\n\n"
                       f"Total reviews uploaded: **{len(df):,} rows**")

            st.write("### 📋 Data Preview")
            st.dataframe(df.head(10))

            # --- Tombol Sentiment ---
            if st.button("🔍 Run Sentiment Prediction"):
                with st.spinner("⏳ Running sentiment analysis... please wait..."):
                    df = st.session_state.uploaded_data
                    df["cleaned_text"] = df[col_name].apply(clean_text)
                    df["stopword_removed"] = df["cleaned_text"].apply(remove_stopwords)
                    df["Predicted_Label"] = df["cleaned_text"].apply(lambda x: label_map[predict_sentiment(x)])

                    st.session_state.sentiment_result = df
                    st.session_state.sentiment_done = True
                    st.session_state.topic_done = False

            if st.session_state.sentiment_done:
                st.success("✅ Sentiment prediction complete! Go to **Tab '📊 Sentiment Analysis'** to view results.")

            # --- Tombol Topic ---
            if st.session_state.sentiment_done:
                if st.button("💡 Run Topic Prediction"):
                    with st.spinner("⏳ Generating topic clusters... please wait..."):
                        df = st.session_state.sentiment_result

                        df_neg = df[df["Predicted_Label"] == "Negatif"][[col_name, "cleaned_text"]].copy()
                        df_net = df[df["Predicted_Label"] == "Netral"][[col_name, "cleaned_text"]].copy()
                        df_pos = df[df["Predicted_Label"] == "Positif"][[col_name, "cleaned_text"]].copy()

                        nlp = load_lemmatizer()

                        if not df_neg.empty:
                            df_neg["stopword_removed"] = df_neg["cleaned_text"].apply(remove_stopwords_topic)
                            df_neg["lemmatized_text"] = df_neg["stopword_removed"].apply(lambda x: lemmatize_text(nlp, x))
                            df_neg["Predicted_Topic"] = df_neg["lemmatized_text"].apply(lambda x: label_map_topic_neg[predict_topic_neg(x)])
                            st.session_state.df_neg_topic = df_neg
                        
                        if not df_net.empty:
                            df_net["stopword_removed"] = df_net["cleaned_text"].apply(remove_stopwords_topic)
                            df_net["lemmatized_text"] = df_net["stopword_removed"].apply(lambda x: lemmatize_text(nlp, x))
                            df_net["Predicted_Topic"] = df_net["lemmatized_text"].apply(lambda x: label_map_topic_net[predict_topic_net(x)])
                            st.session_state.df_net_topic = df_net

                        if not df_pos.empty:
                            df_pos["stopword_removed"] = df_pos["cleaned_text"].apply(remove_stopwords_topic)
                            df_pos["lemmatized_text"] = df_pos["stopword_removed"].apply(lambda x: lemmatize_text(nlp, x))
                            df_pos["Predicted_Topic"] = df_pos["lemmatized_text"].apply(lambda x: label_map_topic_pos[predict_topic_pos(x)])
                            st.session_state.df_pos_topic = df_pos

                        st.session_state.topic_done = True
            else:
                st.button("💡 Run Topic Prediction", disabled=True)
                st.caption("⚠️ Please run sentiment prediction first to enable topic prediction.")

            if st.session_state.topic_done:
                st.success("✅ Topic prediction complete! Go to **Tab '💡 Topic Analysis'** to view results.")

# ==============================
# Tab 2 - Sentiment Results
# ==============================
with tab2:
    if st.session_state.uploaded_data is not None and st.session_state.sentiment_done:
        df = st.session_state.sentiment_result

        # Show table
        st.subheader("📊 Sentiment Prediction Results")
        st.write("Showing top 10 rows. Explore full results by sentiment below ⬇️")
        st.dataframe(df[[col_name, "Predicted_Label"]].head(10))

        # Bar chart distribution
        st.subheader("📈 Sentiment Distribution")
        order = ["Negatif", "Netral", "Positif"]
        sentiment_counts = (
            df["Predicted_Label"]
            .value_counts()
            .reindex(order, fill_value=0)
            .reset_index()
        )
        sentiment_counts.columns = ["Sentiment", "Count"]

        sentiment_counts["Percentage"] = ((sentiment_counts["Count"] / sentiment_counts["Count"].sum() * 100).round()).astype(int)
        sentiment_counts["label"] = sentiment_counts["Percentage"].astype(str) + "% (" + sentiment_counts["Count"].astype(str) + ")"

        bars = alt.Chart(sentiment_counts).mark_bar().encode(
            x=alt.X("Sentiment", sort=order, axis=alt.Axis(labelAngle=0, labelFontSize=18, title=None)),
            y=alt.Y("Count", axis=alt.Axis(labelFontSize=12, title=None, tickMinStep=1)),
            color=alt.Color("Sentiment", scale=alt.Scale(domain=order, range=["#e87b7d", "#68A2E8", "#90E8A6"])),
            tooltip=["Sentiment", "Percentage", "Count"]
        )

        text = bars.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=16,
            fontWeight="bold"
        ).encode(
            text="label:N"
        )

        chart = (bars + text).configure_legend(disable=True)

        st.altair_chart(chart)

        # Word Cloud
        @st.cache_data(show_spinner=False)
        def generate_wordcloud(texts, colormap, max_words):
            text_combined = " ".join(texts)
            
            wc = WordCloud(
                width=800, 
                height=400, 
                background_color="white", 
                colormap=colormap, 
                max_words=max_words
            ).generate(text_combined)
            
            return wc

        # Menyiapkan data per sentimen
        neg_texts = df[df["Predicted_Label"] == "Negatif"]["stopword_removed"].tolist()
        neu_texts = df[df["Predicted_Label"] == "Netral"]["stopword_removed"].tolist()
        pos_texts = df[df["Predicted_Label"] == "Positif"]["stopword_removed"].tolist()

        cmap_map = {
            "Positif": "Greens",
            "Negatif": "Reds",
            "Netral": "Blues"
        }

        # Menampilkan WordCloud per sentimen
        st.subheader("☁️ Sentiment Word Cloud")
        max_words = 50

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, texts, title, cmap in zip(
            axes,
            [neg_texts, neu_texts, pos_texts],
            ["Negatif", "Netral", "Positif"],
            [cmap_map["Negatif"], cmap_map["Netral"], cmap_map["Positif"]]
        ):
            if len(texts) > 0:
                wc = generate_wordcloud(texts, cmap, max_words)
                ax.imshow(wc)
                ax.set_title(title)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_title(title)
                ax.axis("off")

        st.pyplot(fig)

        # --- Dropdown untuk filter review berdasarkan sentiment ---
        st.markdown("### 🔍 Explore Reviews by Sentiment")

        # Get available sentiments from predicted labels
        available_sentiments = df["Predicted_Label"].unique()
        selected_sentiment = st.selectbox("Select sentiment to view results", available_sentiments, index=0)

        # Filter reviews based on the selected sentiment
        filtered_reviews = df[df["Predicted_Label"] == selected_sentiment][[col_name]].dropna()

        if filtered_reviews.empty:
            st.warning("No reviews found for selected sentiment.")
        else:
            st.markdown(f"**Showing all reviews for '{selected_sentiment}'**")
            st.dataframe(filtered_reviews)

        # Download hasil
        st.markdown("---")
        st.markdown("#### 📥 Download Sentiment Prediction Result")
        st.caption("Choose your preferred file format to download the results.")

        df_sentiment_result = df[[col_name, "Predicted_Label"]].copy()
        df_sentiment_result.columns = ["Text", "Predicted_Label"]

        csv_sentiment = df_sentiment_result.to_csv(index=False).encode("utf-8")

        excel_buffer = io.BytesIO()
        df_sentiment_result.to_excel(excel_buffer, index=False, sheet_name="Sentiment_Result")
        excel_buffer.seek(0)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.download_button(
                label="⬇️ Download CSV File",
                data=csv_sentiment,
                file_name="Sentiment_Result.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.download_button(
                label="⬇️ Download Excel File",
                data=excel_buffer,
                file_name="Sentiment_Result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    else:
        st.warning("⚠️ Please upload data or run the sentiment prediction first.")


# ==============================
# 🟠 TAB 3 - Topic Results
# ==============================
with tab3:
    if st.session_state.topic_done:
        st.subheader("💡 Topic Prediction Results")
        
        # Ambil data hasil topik dari session_state
        df_neg = st.session_state.get("df_neg_topic", pd.DataFrame())
        df_net = st.session_state.get("df_net_topic", pd.DataFrame())
        df_pos = st.session_state.get("df_pos_topic", pd.DataFrame())

        # Drop hasil "Outlier"
        if not df_neg.empty:
            df_neg = df_neg[df_neg["Predicted_Topic"] != "Outlier"]
            df_neg["Sentiment"] = "Negatif"

        if not df_net.empty:
            df_net = df_net[df_net["Predicted_Topic"] != "Outlier"]
            df_net["Sentiment"] = "Netral"

        if not df_pos.empty:
            df_pos = df_pos[df_pos["Predicted_Topic"] != "Outlier"]
            df_pos["Sentiment"] = "Positif"

        df_all = pd.concat([df_neg, df_net, df_pos], ignore_index=False).sort_index().reset_index()

        st.write("Showing top 10 rows. Explore all topic clusters per sentiment using the tabs below ⬇️")
        st.dataframe(df_all[[col_name, "Sentiment", "Predicted_Topic"]].head(10))

        st.caption("")
        st.write("Select a sentiment tab below to explore topic clusters.")
        # Buat nested tabs per sentiment
        topic_tabs = st.tabs(["🔴 Negative", "🔵 Neutral", "🟢 Positive"])

        # ======= Tab Negatif =======
        with topic_tabs[0]:
            if df_neg.empty:
                st.info("No negative reviews found.")
            else:
                # Bar chart distribution
                st.subheader("📈 Negative Topic Distribution")
                st.write("📌 The topics below represent negative feedback & issues reported by Transjakarta users")
                order_neg = ["Layanan Transjakarta", "Sistem Pembayaran", "Waktu Tunggu", "Ketersediaan Armada",
                             "Sistem Pengumuman", "Aplikasi Transjakarta", "Fasilitas Halte"]
                topic_counts_neg = (
                    df_neg["Predicted_Topic"]
                    .value_counts()
                    .reindex(order_neg, fill_value=0)
                    .reset_index()
                )

                topic_counts_neg.columns = ["Topic", "Count"]

                topic_counts_neg["Percentage"] = ((topic_counts_neg["Count"] / topic_counts_neg["Count"].sum() * 100).round()).astype(int)
                topic_counts_neg["label"] = topic_counts_neg["Percentage"].astype(str) + "% (" + topic_counts_neg["Count"].astype(str) + ")"

                bars_neg = alt.Chart(topic_counts_neg).mark_bar().encode(
                    x=alt.X(
                        "Topic", 
                        sort=order_neg,
                        axis=alt.Axis(              
                            labelAngle=0,           
                            labelFontSize=12,
                            title=None,
                            labelLimit=0,           
                            labelPadding=15,        
                            labelBaseline="top"    
                        )
                    ),
                    y=alt.Y("Count", axis=alt.Axis(labelFontSize=12, title=None, tickMinStep=1)),
                    color=alt.Color(
                        "Topic", 
                        scale=alt.Scale(
                            domain=order_neg, 
                            range=["#8DAFC8", "#FEB989", "#F49A9D", "#A4D3D0", "#96C498", "#F9DC98", "#DAB7E3"]
                        )
                    ),
                    tooltip=["Topic", "Percentage", "Count"]
                ).properties(
                    width=700,    
                    height=400    
                )

                text_neg = bars_neg.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontSize=16,
                    fontWeight="bold"
                ).encode(
                    text="label:N"
                )

                chart_neg = (bars_neg + text_neg).configure_legend(disable=True)

                st.altair_chart(chart_neg, use_container_width=True)

                # Word Cloud
                topic_color_map_neg = {
                    "Layanan Transjakarta": "#8DAFC8",
                    "Sistem Pembayaran": "#FEB989",
                    "Waktu Tunggu": "#F49A9D",
                    "Ketersediaan Armada": "#A4D3D0",
                    "Sistem Pengumuman": "#96C498",
                    "Aplikasi Transjakarta": "#F9DC98",
                    "Fasilitas Halte": "#DAB7E3"
                }
                
                def generate_wordcloud_neg(texts, topic):
                    color = topic_color_map_neg.get(topic, "#CCCCCC")
                    color_func = get_single_color_func(color)

                    text_combined = " ".join(texts)
                    wc = WordCloud(
                        width=500, 
                        height=250, 
                        background_color="white", 
                        color_func=color_func,                        
                        max_words=30
                    ).generate(text_combined)
                    
                    return wc
                
                st.subheader("🔍 Explore Negative Topics")

                if "Predicted_Topic" in df_neg.columns and len(df_neg) > 0:
                    topics_available = df_neg["Predicted_Topic"].unique().tolist()
                    selected_topic = st.selectbox(
                        "Select a topic to display word cloud and reviews:",
                        options=topics_available,
                        key="topic_neg"
                        )
                    
                    topic_texts = df_neg[df_neg["Predicted_Topic"] == selected_topic]["lemmatized_text"].tolist()

                    st.caption("")
                    st.markdown("#### ☁️ Negative Topic Word Cloud")

                    if len(topic_texts) > 0:
                        wc = generate_wordcloud_neg(topic_texts, selected_topic)
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""<div style='text-align:center; font-weight:600; font-size:20px'>
                                        Topic: {selected_topic} </div>""", unsafe_allow_html=True)
                            st.image(buf, use_container_width=True)
                    else:
                        st.info("No text data available for this topic.")

                    # Explore
                    st.markdown("#### 🔍 Negative Topic Reviews")
                    st.dataframe(df_neg[df_neg["Predicted_Topic"] == selected_topic][col_name])
                else:
                    st.info("No topic prediction results available.")

                # Download hasil
                st.markdown("---")
                st.markdown("#### 📥 Download Negative Topic Prediction Result")
                st.write("Choose your preferred file format to download the results.")

                df_topic_neg_result = df_neg[[col_name, "Predicted_Topic"]].copy()
                df_topic_neg_result.columns = ["Text", "Predicted Topic"]

                csv_topic_neg = df_topic_neg_result.to_csv(index=False).encode("utf-8")

                excel_buffer_neg = io.BytesIO()
                df_topic_neg_result.to_excel(excel_buffer_neg, index=False, sheet_name="Negative_Topic_Result")
                excel_buffer_neg.seek(0)

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_topic_neg,
                        file_name="Negative_Topic_Result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_buffer_neg,
                        file_name="Negative_Topic_Result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        # ======= Tab Netral =======
        with topic_tabs[1]:
            if df_net.empty:
                st.info("No neutral reviews found.")
            else:
                # Bar chart distribution
                st.subheader("📈 Neutral Topic Distribution")
                st.write("📌 The topics below represent feedback & inquiries from Transjakarta users")
                order_net = ["Panduan Rute", "Jadwal Operasional Bus", "Informasi Sistem Pembayaran"]
                topic_counts_net = (
                    df_net["Predicted_Topic"]
                    .value_counts()
                    .reindex(order_net, fill_value=0)
                    .reset_index()
                )
                topic_counts_net.columns = ["Topic", "Count"]

                topic_counts_net["Percentage"] = ((topic_counts_net["Count"] / topic_counts_net["Count"].sum() * 100).round()).astype(int)
                topic_counts_net["label"] = topic_counts_net["Percentage"].astype(str) + "% (" + topic_counts_net["Count"].astype(str) + ")"

                bars_net = alt.Chart(topic_counts_net).mark_bar().encode(
                    x=alt.X(
                        "Topic", 
                        sort=order_net,
                        axis=alt.Axis(              
                            labelAngle=0,          
                            labelFontSize=12,
                            title=None,
                            labelLimit=0,           
                            labelPadding=15,        
                            labelBaseline="top"    
                        )
                    ),
                    y=alt.Y("Count", axis=alt.Axis(labelFontSize=12, title=None, tickMinStep=1)),
                    color=alt.Color("Topic", scale=alt.Scale(domain=order_net, range=["#8DAFC8", "#FEB989", "#DAB7E3"])),
                    tooltip=["Topic", "Percentage", "Count"]
                ).properties(
                    width=700,   
                    height=400   
                )

                text_net = bars_net.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontSize=16,
                    fontWeight="bold"
                ).encode(
                    text="label:N"
                )

                chart_net = (bars_net + text_net).configure_legend(disable=True)

                st.altair_chart(chart_net, use_container_width=True)

                # Word Cloud
                topic_color_map_net = {
                    "Panduan rute": "#8DAFC8",
                    "Jadwal Operasional Bus": "#FEB989",
                    "Informasi Sistem Pembayaran": "#DAB7E3"
                }
                
                def generate_wordcloud_net(texts, topic):
                    color = topic_color_map_net.get(topic, "#CCCCCC")
                    color_func = get_single_color_func(color)

                    text_combined = " ".join(texts)
                    wc = WordCloud(
                        width=500, 
                        height=250, 
                        background_color="white", 
                        color_func=color_func,                        
                        max_words=30
                    ).generate(text_combined)
                    
                    return wc

                st.subheader("🔍 Explore Negative Topics")

                if "Predicted_Topic" in df_net.columns and len(df_net) > 0:
                    topics_available = df_net["Predicted_Topic"].unique().tolist()
                    selected_topic = st.selectbox(
                        "Select a topic to display word cloud and reviews:",
                        options=topics_available,
                        key="topic_net"
                        )
                    
                    topic_texts = df_net[df_net["Predicted_Topic"] == selected_topic]["lemmatized_text"].tolist()

                    st.caption("")
                    st.markdown("#### ☁️ Neutral Topic Word Cloud")

                    if len(topic_texts) > 0:
                        wc = generate_wordcloud_net(topic_texts, selected_topic)
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""<div style='text-align:center; font-weight:600; font-size:20px'>
                                        Topic: {selected_topic} </div>""", unsafe_allow_html=True)
                            st.image(buf, use_container_width=True)
                    else:
                        st.info("No text data available for this topic.")

                    # Explore
                    st.markdown("#### 🔍 Neutral Topic Reviews")
                    st.dataframe(df_net[df_net["Predicted_Topic"] == selected_topic][col_name])

                else:
                    st.info("No topic prediction results available.")

                # Download hasil
                st.markdown("---")
                st.markdown("#### 📥 Download Neutral Topic Prediction Result")
                st.write("Choose your preferred file format to download the results.")

                df_topic_net_result = df_net[[col_name, "Predicted_Topic"]].copy()
                df_topic_net_result.columns = ["Text", "Predicted Topic"]

                csv_topic_net = df_topic_net_result.to_csv(index=False).encode("utf-8")

                excel_buffer_net = io.BytesIO()
                df_topic_net_result.to_excel(excel_buffer_net, index=False, sheet_name="Neutral_Topic_Result")
                excel_buffer_net.seek(0)

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_topic_net,
                        file_name="Neutral_Topic_Result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_buffer_net,
                        file_name="Neutral_Topic_Result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        # ======= Tab Positif =======
        with topic_tabs[2]:
            if df_pos.empty:
                st.info("No positive reviews found.")
            else:
                # Bar chart distribution
                st.subheader("📈 Positive Topic Distribution")
                st.write("📌 The topics below represent positive feedback & appreciation from Transjakarta users")
                order_pos = ["Kenyamanan Transportasi dan Supir", "Apresiasi Pelayanan Petugas",
                             "Pengalaman Positif Layanan Transjakarta", "Ekspresi Pujian", "Ekspansi Rute dan Mobilitas"]
                topic_counts_pos = (
                    df_pos["Predicted_Topic"]
                    .value_counts()
                    .reindex(order_pos, fill_value=0)
                    .reset_index()
                )

                topic_counts_pos.columns = ["Topic", "Count"]

                topic_counts_pos["Percentage"] = ((topic_counts_pos["Count"] / topic_counts_pos["Count"].sum() * 100).round()).astype(int)
                topic_counts_pos["label"] = topic_counts_pos["Percentage"].astype(str) + "% (" + topic_counts_pos["Count"].astype(str) + ")"

                bars_pos = alt.Chart(topic_counts_pos).mark_bar().encode(
                    x=alt.X(
                        "Topic", 
                        sort=order_pos,
                        axis=alt.Axis(              
                            labelAngle=0,           
                            labelFontSize=12,
                            title=None,
                            labelLimit=0,           
                            labelPadding=15,        
                            labelBaseline="top"    
                        )
                    ),
                    y=alt.Y("Count", axis=alt.Axis(labelFontSize=12, title=None, tickMinStep=1)),
                    color=alt.Color("Topic", scale=alt.Scale(domain=order_pos, range=["#F49A9D", "#A4D3D0", "#96C498", "#F9DC98", "#DAB7E3"])),
                    tooltip=["Topic", "Percentage", "Count"]
                ).properties(
                    width=700,    
                    height=400    
                )

                text_pos = bars_pos.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontSize=14,
                    fontWeight="bold"
                ).encode(
                    text="label:N"
                )

                chart_pos = (bars_pos + text_pos).configure_legend(disable=True)

                st.altair_chart(chart_pos, use_container_width=True)

                # Word Cloud
                topic_color_map_pos = {
                    "Kenyamanan Transportasi dan Supir": "#F49A9D",
                    "Apresiasi Pelayanan Petugas": "#A4D3D0",
                    "Pengalaman Positif Layanan Transjakarta": "#96C498",
                    "Ekspresi Pujian": "#F9DC98",
                    "Ekspansi Rute dan Mobilitas": "#DAB7E3"
                }
                
                def generate_wordcloud_pos(texts, topic):
                    color = topic_color_map_pos.get(topic, "#CCCCCC")
                    color_func = get_single_color_func(color)

                    text_combined = " ".join(texts)
                    wc = WordCloud(
                        width=500, 
                        height=250, 
                        background_color="white", 
                        color_func=color_func,                        
                        max_words=30
                    ).generate(text_combined)
                    
                    return wc

                st.subheader("🔍 Explore Positive Topics")

                if "Predicted_Topic" in df_pos.columns and len(df_pos) > 0:
                    topics_available = df_pos["Predicted_Topic"].unique().tolist()
                    selected_topic = st.selectbox(
                        "Select a topic to display word cloud and reviews:",
                        options=topics_available,
                        key="topic_pos"
                        )
                    
                    topic_texts = df_pos[df_pos["Predicted_Topic"] == selected_topic]["lemmatized_text"].tolist()

                    st.caption("")
                    st.markdown("#### ☁️ Positive Topic Word Cloud")

                    if len(topic_texts) > 0:
                        wc = generate_wordcloud_pos(topic_texts, selected_topic)
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""<div style='text-align:center; font-weight:600; font-size:20px'>
                                        Topic: {selected_topic} </div>""", unsafe_allow_html=True)
                            st.image(buf, use_container_width=True)
                    else:
                        st.info("No text data available for this topic.")

                    # Explore
                    st.markdown("#### 🔍 Positive Topic Reviews")
                    st.dataframe(df_pos[df_pos["Predicted_Topic"] == selected_topic][col_name])

                else:
                    st.info("No topic prediction results available.")

                # Download hasil
                st.markdown("---")
                st.markdown("#### 📥 Download Positive Topic Prediction Result")
                st.write("Choose your preferred file format to download the results.")

                df_topic_pos_result = df_pos[[col_name, "Predicted_Topic"]].copy()
                df_topic_pos_result.columns = ["Text", "Predicted Topic"]

                csv_topic_pos = df_topic_pos_result.to_csv(index=False).encode("utf-8")

                excel_buffer_pos = io.BytesIO()
                df_topic_pos_result.to_excel(excel_buffer_pos, index=False, sheet_name="Positive_Topic_Result")
                excel_buffer_pos.seek(0)

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_topic_pos,
                        file_name="Positive_Topic_Result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_buffer_pos,
                        file_name="Positive_Topic_Result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

    else:
        st.warning("⚠️ Please run the topic prediction first.")
