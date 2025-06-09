import streamlit as st
import pandas as pd
import gspread
import re
from bs4 import BeautifulSoup
import requests
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- 1. Streamlit Page Configuration (MUST BE THE FIRST ST. COMMAND) ---
st.set_page_config(
    page_title="Butler Coffee Lab – Flavor Match App",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# --- 2. Configuration & Initialization ---
GOOGLE_SHEET_ID = "1VBnG4kfGOUN3iVH1n14qOUnzivhiU_SsOclCAcWkFI8"

try:
    openai_api_key = st.secrets["openai"]["api_key"]
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("Looks like your OpenAI API key isn't set up correctly in Streamlit Secrets. Please ensure 'open_ai_key' is present.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# --- 3. Helper Functions ---

@st.cache_data(ttl=3600)
def load_data_from_google_sheets():
    try:
        if "google_service_account" in st.secrets:
            gc = gspread.service_account_from_dict(st.secrets["google_service_account"])
        else:
            st.error("Google service account secrets not found. Please configure 'google_service_account' in your Streamlit secrets.")
            st.stop()

        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        worksheet = sh.worksheet("Sheet1")

        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        df.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns]

        required_cols = ['name', 'category', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers (e.g., 'Category' becomes 'category').")
                return pd.DataFrame()

        if 'brew_method' not in df.columns:
            df['brew_method'] = ''
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''

        return df
    except Exception as e:
        st.error(f"Uh oh! Couldn't load data from Google Sheets. Error: {e}")
        st.info("Double-check your Google Sheet ID, and ensure the service account email has Viewer access to the sheet.")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*24)
def get_embeddings(texts):
    valid_texts = [str(t) for t in texts if pd.notna(t) and t.strip() != '']
    if not valid_texts:
        return []
    embeddings = embedder.encode(valid_texts, convert_to_tensor=True)
    return [embeddings[i] for i in range(len(embeddings))]

@st.cache_data(ttl=3600)
def extract_flavor_tags(descriptions):
    prompts = [
        {"role": "system", "content": "Extract 3 to 5 unique flavor notes from the product description. Return them as a comma-separated list."},
    ]
    tags = []
    for desc in descriptions:
        if not desc.strip():
            tags.append("")
            continue
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompts + [{"role": "user", "content": desc}],
                max_tokens=50,
                temperature=0.7
            )
            tag_text = response.choices[0].message.content.strip()
            tags.append(tag_text)
        except:
            tags.append("")
    return tags

# --- 4. Main App Logic ---
st.title("\u2615\ufe0f Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

# Ensure image_url column exists
if 'image_url' not in df_products.columns:
    df_products['image_url'] = ''

# Add 'specific_flavors' column from extracted flavor_tags
if 'specific_flavors' not in df_products.columns:
        df_products['specific_flavors'] = extract_flavor_tags(df_products['long_description'].fillna(''))

if not df_products.empty:
    df_products['flavor_tags'] = df_products['specific_flavors']

    flavor_suggestions = sorted(set(
        tag.strip()
        for tag_list in df_products['flavor_tags'].str.split(',')
        if isinstance(tag_list, list)
        for tag in tag_list
        if tag.strip()
    ))

    valid_descriptions = df_products['flavor_tags'].fillna('').tolist()
    cleaned_descriptions = [desc if desc.strip() != '' else 'flavor unknown' for desc in valid_descriptions]
    all_embeddings = get_embeddings(cleaned_descriptions)

    if len(all_embeddings) != len(cleaned_descriptions):
        st.error("Mismatch in embeddings. Please check data format.")
        st.stop()

    df_products['flavor_tag_embedding'] = all_embeddings

    st.sidebar.header("Tell us your preferences!")

    with st.sidebar.form("flavor_form"):
        st.markdown("**1. What are you in the mood for?**")
        drink_type = st.radio("Drink Type", ["Coffee", "Tea"], horizontal=True, index=0)

        st.markdown("**2. How do you brew?**")
        brew_method = st.multiselect("Brew Method", ["Pods", "Ground", "Whole Bean"], default=["Ground"])

        st.markdown("**3. Tell us about your taste preferences**")
        freeform_input = st.text_input("Describe what you like (e.g., 'I like sweet and cozy drinks')")

        suggested_tags = []
        if freeform_input.strip():
            try:
                ai_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Based on a user's taste description, return 3-5 matching flavor notes from this list only: " + ', '.join(flavor_suggestions)},
                        {"role": "user", "content": freeform_input}
                    ],
                    max_tokens=50,
                    temperature=0.7
                )
                result = ai_response.choices[0].message.content.strip()
                suggested_tags = [tag.strip() for tag in result.split(',') if tag.strip() in flavor_suggestions]
                if suggested_tags:
                    st.markdown("Suggested flavor tags based on what you wrote:")
                    st.write(suggested_tags)
            except Exception as e:
                st.warning(f"AI suggestion failed: {e}")
        flavor_input = st.multiselect("Select flavor notes you'd like to include:", options=flavor_suggestions, default=suggested_tags)

        roast_preference = "No preference"
        if drink_type == "Coffee":
            st.markdown("**4. How do you like your coffee roasted?**")
            roast_preference = st.radio("Roast/Intensity", ["Light", "Medium", "Dark", "No preference"], horizontal=True, index=3)

        st.markdown("**5. Feeling adventurous?**")
        surprise_me = st.checkbox("Surprise Me!")

        submitted = st.form_submit_button("Find My Perfect Match!")

    if submitted:
        # Use AI to classify each product into a broader flavor category
        if 'flavor_category' not in df_products.columns:
            df_products['flavor_category'] = 'General Favorites'

        with st.spinner("Brewing up your perfect recommendations..."):
            filtered_products = df_products.copy()

            temp = filtered_products[filtered_products['category'].str.contains(drink_type, case=False, na=False)]
            if not temp.empty:
                filtered_products = temp

            if brew_method:
                temp = filtered_products[filtered_products['brew_method'].fillna('').apply(lambda x: any(b.lower() in (x + ' ' + filtered_products.get('name', '')).lower() for b in brew_method))]
                if not temp.empty:
                    filtered_products = temp

            if drink_type == "Coffee" and roast_preference != "No preference":
                temp = filtered_products[filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)]
                if not temp.empty:
                    filtered_products = temp

            recommendations = pd.DataFrame()
            top_categories = df_products['flavor_category'].value_counts().head(3).index.tolist()
            if not flavor_input and not surprise_me:
                st.info("Choose a flavor category to explore:")
                chosen_category = st.selectbox("Top Flavor Categories:", options=top_categories)
                recommendations = df_products[df_products['flavor_category'] == chosen_category].head(5)

            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'.")
            elif flavor_input:
                products_for_similarity = filtered_products.copy()
                products_for_similarity['flavor_overlap'] = products_for_similarity['flavor_tags'].apply(lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else [])))
                matched = products_for_similarity[products_for_similarity['flavor_overlap'] > 0].copy()
                recommendations = matched.sort_values(by='flavor_overlap', ascending=False).head(5)

                if recommendations.empty:
                    # Use AI to suggest adjacent tags
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that enhances vague or short flavor preferences by suggesting related tags."},
                                {"role": "user", "content": f"These are the flavor notes the user selected: {', '.join(flavor_input)}. Suggest 3-5 similar or adjacent flavor notes that appear in this list: {', '.join(flavor_suggestions)}."}
                            ],
                            max_tokens=100,
                            temperature=0.7
                        )
                        new_tags = response.choices[0].message.content.strip().split(',')
                        new_tags = [tag.strip() for tag in new_tags if tag.strip() in flavor_suggestions and tag.strip() not in flavor_input]
                        if new_tags:
                            st.session_state.setdefault('ai_flavor_suggestion', None)
                            st.session_state.setdefault('flavor_response', None)
                            if st.session_state['ai_flavor_suggestion'] != new_tags:
                                st.session_state['ai_flavor_suggestion'] = new_tags
                                st.session_state['flavor_response'] = None

                            if st.session_state['flavor_response'] is None:
                                st.markdown(f"It sounds like you enjoy flavors like {', '.join(flavor_input)}. You might also like: {', '.join(new_tags)}")
        else:
            st.info("We're showing you a few of our favorite brews based on general preferences — try adjusting your flavor selections or just click 'Surprise Me' next time!")
else:
    st.error("There was a problem loading the product data. Please check the app configuration and try again later.")
