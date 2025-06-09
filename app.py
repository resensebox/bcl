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
        if 'image_url' not in df.columns: # Ensure image_url column exists
            df['image_url'] = ''

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
        except Exception as e:
            tags.append("")
    return tags

# --- 4. Main App Logic ---
st.title("\u2615\ufe0f Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

# Add 'specific_flavors' column from extracted flavor_tags if it doesn't exist
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

    st.sidebar.header("Let’s build your perfect brew together!")

    # Define drink_type here
    drink_type = st.sidebar.selectbox("Step 0: What are you looking for?", ["Coffee", "Tea", "Other"], index=0)

    step_description = st.sidebar.text_area("Step 1: Describe your ideal coffee moment (e.g., 'I want something cozy for rainy mornings', 'a bold flavor to start my day', or 'smooth and sweet like dessert')", height=100)
    recommended_tags = []
    if step_description.strip():
        try:
            response = client.chat.com.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You’re a flavor concierge helping someone describe what they like to drink. Based on their answer, suggest 3-5 flavor notes from this list only: " + ', '.join(flavor_suggestions)},
                    {"role": "user", "content": step_description}
                ],
                max_tokens=50,
                temperature=0.7
            )
            tag_text = response.choices[0].message.content.strip()
            recommended_tags = [t.strip() for t in tag_text.split(',') if t.strip() in flavor_suggestions]
            if recommended_tags:
                st.sidebar.markdown("These sound like a fit. Do you agree?")
                agree = st.sidebar.radio("Are these on point?", ["Yes", "No"], key="ai_tags_agree") # Added key for uniqueness
                if agree == "Yes":
                    st.sidebar.success("Awesome! We’ll use these to find your best match.")
                if agree == "No":
                    st.sidebar.markdown("Okay! Pick a few flavor notes below or rewrite your description.")
                    recommended_tags = []
        except Exception as e:
            st.sidebar.warning("We had trouble understanding your flavor style. Try typing again.")

    st.sidebar.markdown("Step 2: Choose your brew method")
    brew_method = st.sidebar.multiselect("Brew Method", ["Pods", "Ground", "Whole Bean"], default=["Ground"])

    st.sidebar.markdown("Step 3: Select your flavor notes (you can mix & match)")
    flavor_input = st.sidebar.multiselect("Flavor Notes", options=flavor_suggestions, default=recommended_tags)

    st.sidebar.markdown("Step 4: Roast Preference")
    roast_preference = "No preference"
    if st.sidebar.checkbox("Are you a coffee drinker?"): # This condition might hide roast preference for tea drinkers
        roast_preference = st.sidebar.radio("Roast/Intensity", ["Light", "Medium", "Dark", "No preference"], horizontal=True, index=3)

    st.sidebar.markdown("Step 5: Feeling adventurous?")
    surprise_me = st.sidebar.checkbox("Surprise Me!")

    submitted = st.sidebar.button("Find My Perfect Match!")

    if submitted:
        # Use AI to classify each product into a broader flavor category (This part of the code is not fully utilized later for filtering)
        if 'flavor_category' not in df_products.columns:
            df_products['flavor_category'] = 'General Favorites' # Default if not assigned

        with st.spinner("Brewing up your perfect recommendations..."):
            filtered_products = df_products.copy()

            # Filter by drink type (category)
            if drink_type:
                temp = filtered_products[filtered_products['category'].str.contains(drink_type, case=False, na=False)]
                if not temp.empty:
                    filtered_products = temp
                else:
                    st.warning(f"No products found for '{drink_type}' category. Showing results from all categories.")
                    filtered_products = df_products.copy() # Reset filter if no matches

            # Filter by brew method
            if brew_method:
                temp = filtered_products[
                    filtered_products.apply(
                        lambda row: any(
                            b.lower() in (str(row['brew_method']) + ' ' + str(row['name'])).lower()
                            for b in brew_method
                        ),
                        axis=1
                    )
                ]
                if not temp.empty:
                    filtered_products = temp
                else:
                    st.warning("No products found matching your selected brew method. Showing results ignoring brew method.")
                    # If no products match brew method, don't filter by it.
                    # This implies you want to fall back to broader results, so we might need to be careful here.
                    # For now, I'll reset `filtered_products` to what it was before brew method filter if nothing matches.
                    # A better approach might be to store a `potential_products` and `final_products`
                    pass # Keep the previous filtered_products if no match for brew method

            # Filter by roast preference if Coffee is selected
            if drink_type == "Coffee" and roast_preference != "No preference":
                temp = filtered_products[filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)]
                if not temp.empty:
                    filtered_products = temp
                else:
                    st.warning(f"No coffee found with '{roast_preference}' roast level. Showing all roasts for coffee.")
                    # Keep previous filtered_products if no match for roast level
                    pass


            recommendations = pd.DataFrame()
            top_categories = df_products['flavor_category'].value_counts().head(3).index.tolist()

            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'. Showing general favorites.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42) # Fallback to general products

            elif flavor_input:
                products_for_similarity = filtered_products.copy()
                products_for_similarity['flavor_overlap'] = products_for_similarity['flavor_tags'].apply(
                    lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else []))
                )
                matched = products_for_similarity[products_for_similarity['flavor_overlap'] > 0].copy()

                if matched.empty:
                    st.info("No exact flavor match found. Trying to find similar flavors...")
                    try:
                        ai_adjacent = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant. Based on a list of flavors, suggest 3-5 similar or adjacent flavor notes from this list only: " + ', '.join(flavor_suggestions)},
                                {"role": "user", "content": ', '.join(flavor_input)}
                            ],
                            max_tokens=50,
                            temperature=0.7
                        )
                        new_tags = [tag.strip() for tag in ai_adjacent.choices[0].message.content.strip().split(',') if tag.strip() in flavor_suggestions]
                        if new_tags:
                            st.markdown(f"It sounds like you enjoy flavors like {', '.join(flavor_input)}. You might also like: {', '.join(new_tags)}")
                            # Append new_tags to flavor_input for a broader search
                            flavor_input.extend(new_tags)
                            # Re-calculate overlap with expanded flavor_input
                            products_for_similarity['flavor_overlap'] = products_for_similarity['flavor_tags'].apply(
                                lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else []))
                            )
                            matched = products_for_similarity[products_for_similarity['flavor_overlap'] > 0]
                    except Exception as e:
                        st.warning(f"AI fallback flavor suggestions failed: {e}")

                if not matched.empty:
                    recommendations = matched.sort_values(by='flavor_overlap', ascending=False).head(5)
                else:
                    st.info("No products found matching your selected flavor notes, even with AI suggestions. Showing general favorites.")
                    # Fallback to general favorites if no flavor match at all
                    # This part needs to pick from `filtered_products` if possible, otherwise `df_products`
                    if not filtered_products.empty:
                        recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                    else:
                        recommendations = df_products.sample(min(5, len(df_products)), random_state=42)


            if recommendations.empty and not (flavor_input or surprise_me):
                st.markdown("""
                #### Not sure where to start?
                Here are some categories you might like:
                - **Cozy & Comforting** – warm, smooth, nostalgic
                - **Bold & Robust** – dark, strong, earthy
                - **Sweet & Treat-like** – chocolate, caramel, vanilla
                - **Nutty & Spiced** – hazelnut, cinnamon, pecan
                Choose one and we’ll guide you through flavors you’ll love.
                """)
                st.info("Choose a flavor category to explore:")
                chosen_category = st.selectbox("Top Flavor Categories:", options=top_categories)
                recommendations = df_products[df_products['flavor_category'] == chosen_category].head(5)

            if not recommendations.empty:
                st.markdown("### Your Matches:")
                for _, row in recommendations.iterrows():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if row['image_url'] and isinstance(row['image_url'], str) and row['image_url'].startswith("http"):
                            try:
                                st.image(row['image_url'], use_container_width=True)
                            except:
                                st.caption("Invalid image URL")
                        else:
                            st.caption("No image available")
                    with col2:
                        st.subheader(row['name'])
                        st.caption(f"{row['short_description']} — ${row['price']}")
                        # Display specific flavors if available
                        if row['specific_flavors']:
                            st.markdown(f"**Flavor Notes:** {row['specific_flavors']}")
                        st.markdown(f"[View Product]({row['bcl_website_link']})")
            else:
                st.info("We couldn't find any perfect matches based on your selections. Please try adjusting your preferences!")

else:
    st.error("There was a problem loading the product data. Please check the app configuration and try again later.")
