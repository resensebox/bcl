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
    st.error("Looks like your OpenAI API key isn't set up correctly in Streamlit Secrets. Please ensure 'openai.api_key' is present in your Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner="Loading AI model...")
def load_embedding_model():
    """Loads the Sentence Transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# --- 3. Helper Functions ---

@st.cache_data(ttl=3600, show_spinner="Loading product data...")
def load_data_from_google_sheets():
    """Loads product data from Google Sheets."""
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

        # Sanitize column names
        df.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns]

        required_cols = ['name', 'category', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers (e.g., 'Category' becomes 'category').")
                return pd.DataFrame()

        # Ensure essential columns exist, add if missing
        if 'brew_method' not in df.columns:
            df['brew_method'] = ''
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''
        if 'image_url' not in df.columns:
            df['image_url'] = ''

        return df
    except Exception as e:
        st.error(f"Uh oh! Couldn't load data from Google Sheets. Error: {e}")
        st.info("Double-check your Google Sheet ID, and ensure the service account email has Viewer access to the sheet.")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*24, show_spinner="Generating flavor embeddings...")
def get_embeddings(texts):
    """Generates embeddings for a list of texts."""
    valid_texts = [str(t) for t in texts if pd.notna(t) and t.strip() != '']
    if not valid_texts:
        return []
    embeddings = embedder.encode(valid_texts, convert_to_tensor=True)
    return [embeddings[i] for i in range(len(embeddings))]

@st.cache_data(ttl=3600, show_spinner="Extracting AI flavor tags...")
def extract_flavor_tags(descriptions):
    """Extracts flavor tags from descriptions using OpenAI."""
    tags = []
    for i, desc in enumerate(descriptions):
        if not desc or not desc.strip():
            tags.append("")
            continue
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract 3 to 5 unique, concise flavor notes from the product description. Return them as a comma-separated list (e.g., 'chocolate, caramel, nutty')."},
                    {"role": "user", "content": desc}
                ],
                max_tokens=50,
                temperature=0.7
            )
            tag_text = response.choices[0].message.content.strip()
            tags.append(tag_text)
        except Exception as e:
            # st.warning(f"Error extracting tags for item {i}: {e}") # Debugging
            tags.append("") # Append empty string on error
    return tags

# --- 4. Main App Logic ---
st.title("\u2615\ufe0f Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

if not df_products.empty:
    # Pre-process flavor tags only once
    if 'specific_flavors' not in df_products.columns or df_products['specific_flavors'].isnull().any():
        with st.spinner("Analyzing product flavors with AI (this may take a moment on first run)..."):
            df_products['specific_flavors'] = extract_flavor_tags(df_products['long_description'].fillna(''))
            # Filter out empty or whitespace-only tags
            df_products['specific_flavors'] = df_products['specific_flavors'].apply(
                lambda x: ', '.join([tag.strip() for tag in x.split(',') if tag.strip()]) if isinstance(x, str) else ''
            )

    df_products['flavor_tags'] = df_products['specific_flavors'] # Use this column for consistency

    # Create a comprehensive list of all unique flavor suggestions
    all_flavor_suggestions = sorted(set(
        tag.strip()
        for tag_list in df_products['flavor_tags'].str.split(',')
        if isinstance(tag_list, list)
        for tag in tag_list
        if tag.strip()
    ))

    # Generate embeddings for flavor tags (only for non-empty tags)
    valid_flavor_texts = df_products['flavor_tags'].apply(lambda x: x if x.strip() else 'flavor unknown').tolist()
    all_flavor_embeddings = get_embeddings(valid_flavor_texts)
    df_products['flavor_tag_embedding'] = all_flavor_embeddings

    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("Let’s find your perfect brew!")

        st.markdown("---")
        st.subheader("Step 1: What are you looking for?")
        drink_type = st.radio("Choose a category:", ["Coffee", "Tea", "Other"], key="drink_type_radio")

        st.markdown("---")
        st.subheader("Step 2: How do you brew?")
        uses_keurig = st.checkbox("Do you use a Keurig (for pods)?", key="uses_keurig_checkbox")
        brew_method_options = []
        if uses_keurig:
            brew_method_options.append("Pods")
        else:
            brew_method_options = st.multiselect(
                "Select your brew method(s):",
                ["Ground", "Whole Bean"],
                default=["Ground"] if "Ground" in df_products['brew_method'].str.lower().unique() else [],
                key="other_brew_method_multiselect"
            )

        st.markdown("---")
        st.subheader("Step 3: Tell us about your ideal flavor.")

        # Initialize session state for AI suggested tags
        if 'ai_suggested_tags' not in st.session_state:
            st.session_state.ai_suggested_tags = []

        description_input = st.text_area(
            "Describe your ideal coffee/tea moment (e.g., 'I want something cozy for rainy mornings', 'a bold flavor to start my day', 'smooth and sweet like dessert')",
            height=100, key="flavor_description_input"
        )

        if description_input.strip():
            with st.spinner("Understanding your style..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a flavor concierge. Based on the user's description, suggest 3-5 relevant flavor notes from this specific list only: {', '.join(all_flavor_suggestions)}. If no direct match, suggest the closest ones. Return only the comma-separated flavor notes."},
                            {"role": "user", "content": description_input}
                        ],
                        max_tokens=50,
                        temperature=0.7
                    )
                    ai_tags_raw = response.choices[0].message.content.strip()
                    # Filter to ensure only valid tags from our list are included
                    ai_suggested_tags_filtered = [
                        t.strip() for t t in ai_tags_raw.split(',') if t.strip() in all_flavor_suggestions
                    ]

                    if ai_suggested_tags_filtered:
                        st.session_state.ai_suggested_tags = ai_suggested_tags_filtered
                        st.markdown(f"**We think you might like these flavors:** {', '.join(ai_suggested_tags_filtered)}")
                        agree_to_ai_tags = st.radio(
                            "Are these on point?",
                            ["Yes, use these", "No, I'll pick"],
                            key="ai_tags_agreement"
                        )
                        if agree_to_ai_tags == "No, I'll pick":
                            st.session_state.ai_suggested_tags = [] # Clear AI tags if user wants to pick
                    else:
                        st.warning("We had trouble understanding your flavor style. Please try typing again or pick manually below.")
                        st.session_state.ai_suggested_tags = [] # Clear AI tags if AI fails
                except Exception as e:
                    st.warning(f"AI flavor suggestion failed: {e}. Please try typing again or pick manually below.")
                    st.session_state.ai_suggested_tags = []

        # Flavor multi-select (default to AI suggested or empty if AI failed/rejected)
        flavor_input = st.multiselect(
            "Or, select specific flavor notes:",
            options=all_flavor_suggestions,
            default=st.session_state.ai_suggested_tags,
            key="manual_flavor_select"
        )

        # Option for users who don't know
        if not flavor_input and not description_input.strip():
            if st.checkbox("I'm not sure about flavors, show me popular types!", key="not_sure_flavors"):
                st.info("Okay, we'll suggest some popular general categories for you!")
                st.session_state.explore_categories = True
            else:
                st.session_state.explore_categories = False
        else:
            st.session_state.explore_categories = False # Reset if they start inputting

        st.markdown("---")
        st.subheader("Step 4: Roast Preference (for Coffee)")
        roast_preference = "No preference"
        if drink_type == "Coffee": # Only show roast preference for coffee
            roast_preference = st.radio(
                "Roast/Intensity:",
                ["Light", "Medium", "Dark", "No preference"],
                horizontal=True,
                index=3,
                key="roast_preference_radio"
            )
        else:
            st.info("Roast preference is typically for coffee. Select 'Coffee' above to see this option.")

        st.markdown("---")
        st.subheader("Step 5: Ready to discover?")
        surprise_me = st.checkbox("Surprise Me!", key="surprise_me_checkbox")
        submitted = st.button("Find My Perfect Match!", type="primary", key="find_match_button")

    # --- Main Content Area for Results ---
    if submitted:
        with st.spinner("Finding your perfect matches..."):
            # Start with all products
            current_filtered_products = df_products.copy()

            # 1. Filter by Drink Type (Category)
            if drink_type:
                initial_filter_count = len(current_filtered_products)
                temp_df = current_filtered_products[current_filtered_products['category'].str.contains(drink_type, case=False, na=False)]
                if not temp_df.empty:
                    current_filtered_products = temp_df
                else:
                    st.warning(f"No products found specifically for '{drink_type}'. Showing results from all categories for further filtering.")
                    # If category filter yields nothing, we keep all products to apply other filters
                    current_filtered_products = df_products.copy()

            # 2. Filter by Brew Method
            if brew_method_options: # Check if any brew methods were selected
                initial_filter_count = len(current_filtered_products)
                brew_method_lower = [b.lower() for b in brew_method_options]
                temp_df = current_filtered_products[
                    current_filtered_products.apply(
                        lambda row: any(
                            b_m in (str(row['brew_method']).lower() + ' ' + str(row['name']).lower()) # Check brew method column AND name for "pods"
                            for b_m in brew_method_lower
                        ),
                        axis=1
                    )
                ]
                if not temp_df.empty:
                    current_filtered_products = temp_df
                else:
                    st.warning("No products found matching your selected brew method(s). Broadening search by ignoring brew method.")
                    # Revert to products before brew method filter if nothing matches
                    # This logic should be more robust. For simplicity, we'll just not apply this filter
                    pass # Keep the `current_filtered_products` as they were before this filter if it fails

            # 3. Filter by Roast Preference (only for Coffee)
            if drink_type == "Coffee" and roast_preference != "No preference":
                initial_filter_count = len(current_filtered_products)
                temp_df = current_filtered_products[current_filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)]
                if not temp_df.empty:
                    current_filtered_products = temp_df
                else:
                    st.warning(f"No coffee found with '{roast_preference}' roast level. Broadening search by ignoring roast level for coffee.")
                    pass # Keep the `current_filtered_products` as they were before this filter if it fails

            # Now, apply flavor matching or surprise logic
            recommendations = pd.DataFrame()

            if surprise_me:
                if not current_filtered_products.empty:
                    recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'. Showing top general favorites.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42) # Fallback to general products

            elif flavor_input:
                products_for_similarity = current_filtered_products.copy()

                # Calculate flavor overlap score
                products_for_similarity['flavor_overlap_score'] = products_for_similarity['flavor_tags'].apply(
                    lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else []))
                )

                # Sort by overlap score and get initial recommendations
                recommendations = products_for_similarity[products_for_similarity['flavor_overlap_score'] > 0].sort_values(
                    by='flavor_overlap_score', ascending=False
                ).head(5)

                # If no direct flavor overlap, try semantic similarity (using embeddings)
                if recommendations.empty:
                    st.info("No direct flavor matches found. Searching for similar flavor profiles...")
                    try:
                        user_input_embedding = embedder.encode(", ".join(flavor_input), convert_to_tensor=True)
                        # Filter out products with empty embeddings (should be 'flavor unknown' due to earlier cleaning)
                        valid_embeddings_df = products_for_similarity[
                            products_for_similarity['flavor_tag_embedding'].apply(lambda x: x is not None and len(x) > 0)
                        ].copy()

                        if not valid_embeddings_df.empty:
                            # Convert list of tensors to a single tensor for cosine similarity
                            product_embeddings_tensor = util.cat_embeddings_to_tensor(valid_embeddings_df['flavor_tag_embedding'].tolist())
                            cosine_scores = util.cos_sim(user_input_embedding, product_embeddings_tensor)[0]
                            valid_embeddings_df['similarity_score'] = cosine_scores.cpu().numpy()

                            # Sort by similarity and get top N
                            recommendations = valid_embeddings_df.sort_values(by='similarity_score', ascending=False).head(5)
                            if not recommendations.empty:
                                st.success("Found some great matches with similar flavor profiles!")
                            else:
                                st.warning("Even with AI, we couldn't find close flavor matches. Showing some popular options instead.")
                                recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)
                        else:
                            st.warning("No products with valid flavor embeddings to compare. Showing popular options.")
                            recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)


                    except Exception as e:
                        st.error(f"Error during semantic similarity search: {e}. Showing popular options.")
                        recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)

            # Fallback if no specific selection and "not sure" is checked
            elif st.session_state.explore_categories and not surprise_me:
                st.markdown("### Explore by Flavor Category:")
                top_categories = df_products['flavor_category'].value_counts().head(5).index.tolist()
                # Use a dummy assignment for now if 'flavor_category' not pre-populated
                if 'flavor_category' not in df_products.columns or df_products['flavor_category'].isnull().all():
                     # Simple rule-based categorization if AI categorization isn't implemented elsewhere
                    def categorize_flavor(row):
                        if 'chocolate' in row['specific_flavors'].lower() or 'caramel' in row['specific_flavors'].lower() or 'vanilla' in row['specific_flavors'].lower():
                            return 'Sweet & Dessert-like'
                        elif 'nutty' in row['specific_flavors'].lower() or 'pecan' in row['specific_flavors'].lower() or 'almond' in row['specific_flavors'].lower():
                            return 'Nutty & Rich'
                        elif 'berry' in row['specific_flavors'].lower() or 'citrus' in row['specific_flavors'].lower() or 'floral' in row['specific_flavors'].lower():
                            return 'Fruity & Floral'
                        elif 'dark chocolate' in row['specific_flavors'].lower() or 'smoky' in row['specific_flavors'].lower() or 'bold' in row['short_description'].lower():
                            return 'Bold & Intense'
                        elif 'smooth' in row['specific_flavors'].lower() or 'creamy' in row['specific_flavors'].lower() or 'mild' in row['short_description'].lower():
                            return 'Smooth & Balanced'
                        return 'General Favorites'
                    df_products['flavor_category'] = df_products.apply(categorize_flavor, axis=1)
                    top_categories = df_products['flavor_category'].value_counts().head(5).index.tolist()


                chosen_category = st.selectbox("Select a popular flavor category:", options=top_categories, key="chosen_category_select")
                recommendations = df_products[df_products['flavor_category'] == chosen_category].head(5)
                st.info(f"Showing popular products in '{chosen_category}' category.")

            else: # No specific input, no surprise, no "not sure" selected
                st.info("Please describe your ideal coffee, select flavor notes, or check 'Surprise Me!' to get recommendations.")
                # Show general favorites if no input
                recommendations = df_products.sample(min(5, len(df_products)), random_state=42)


        # --- Display Recommendations ---
        if not recommendations.empty:
            st.markdown("### Your Top Matches:")
            for idx, row in recommendations.iterrows():
                st.markdown("---") # Separator between products
                col1, col2 = st.columns([1, 2])
                with col1:
                    if row['image_url'] and isinstance(row['image_url'], str) and row['image_url'].startswith("http"):
                        try:
                            st.image(row['image_url'], use_container_width=True)
                        except Exception: # Catch any error during image loading
                            st.caption("Image not available.")
                    else:
                        st.caption("No image available.")
                with col2:
                    st.subheader(row['name'])
                    st.markdown(f"**Category:** {row['category']}")
                    if row['brew_method'].strip():
                        st.markdown(f"**Brew Method:** {row['brew_method']}")
                    if row['roast_level'].strip():
                        st.markdown(f"**Roast Level:** {row['roast_level']}")
                    st.write(f"*{row['short_description']}*")
                    st.markdown(f"**Price:** ${row['price']:.2f}") # Format price
                    if row['specific_flavors']:
                        st.markdown(f"**Flavor Notes:** {row['specific_flavors']}")
                    st.markdown(f"[**View Product on Butler Coffee Lab**]({row['bcl_website_link']})")
        else:
            st.error("Apologies! We couldn't find any products that perfectly match all your selections. Please try adjusting your preferences, selecting fewer filters, or letting us 'Surprise Me!'")

else:
    st.error("There was a problem loading the product data. Please check the app configuration and try again later.")
    st.info("Make sure your Google Sheet is published to the web and the service account has viewer access.")
