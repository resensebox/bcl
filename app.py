import streamlit as st
import pandas as pd
import gspread
import re
import requests
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

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

        # Added 'grind_type' to required columns
        required_cols = ['name', 'category', 'short_description', 'long_description', 'price', 'bcl_website_link', 'grind_type']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers (e.g., 'Category' becomes 'category', 'Grind Type' becomes 'grind_type').")
                return pd.DataFrame()

        # Ensure essential columns exist, add if missing
        if 'brew_method' not in df.columns:
            df['brew_method'] = '' # Keep for compatibility, though grind_type is preferred
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''
        if 'image_url' not in df.columns:
            df['image_url'] = ''

        # Convert relevant columns to string type to prevent errors during processing
        for col in ['name', 'category', 'short_description', 'long_description', 'brew_method', 'roast_level', 'bcl_website_link', 'image_url', 'grind_type', 'price']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else '')
            else:
                df[col] = '' # Add missing columns as empty strings


        # Attempt to convert price to numeric, coercing errors
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)

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
        # Return zero vectors with correct dimension for the number of input texts
        return [torch.zeros(embedder.get_sentence_embedding_dimension()) for _ in range(len(texts))]
    
    embeddings = embedder.encode(valid_texts, convert_to_tensor=True)
    
    result_embeddings = []
    text_idx = 0
    for t in texts:
        if pd.notna(t) and str(t).strip() != '':
            result_embeddings.append(embeddings[text_idx])
            text_idx += 1
        else:
            result_embeddings.append(torch.zeros(embedder.get_sentence_embedding_dimension()))
            
    return result_embeddings


@st.cache_data(ttl=3600, show_spinner="Extracting AI flavor tags...")
def extract_flavor_tags(data_series_name, data_series_long_desc):
    """Extracts flavor tags from descriptions (and names) using OpenAI."""
    tags = []
    # Combine name and long_description for better flavor extraction context
    combined_descriptions = [
        f"{name}. {desc}" for name, desc in zip(data_series_name.fillna(''), data_series_long_desc.fillna(''))
    ]

    for i, desc in enumerate(combined_descriptions):
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
            tags.append("") # Append empty string on error
    return tags

# --- 4. Main App Logic ---
st.title("\u2615\ufe0f Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

if not df_products.empty:
    # Pre-process flavor tags only once, considering both name and long_description
    if 'specific_flavors' not in df_products.columns or \
       df_products['specific_flavors'].isnull().any() or \
       (df_products['specific_flavors'] == '').any():
        with st.spinner("Analyzing product flavors with AI (this may take a moment on first run or if data updated)..."):
            df_products['specific_flavors'] = extract_flavor_tags(
                df_products['name'], df_products['long_description']
            )
            df_products['specific_flavors'] = df_products['specific_flavors'].apply(
                lambda x: ', '.join([tag.strip() for tag in x.split(',') if tag.strip()]) if isinstance(x, str) else ''
            )

    df_products['flavor_tags'] = df_products['specific_flavors']

    all_flavor_suggestions = sorted(list(set(
        tag.strip()
        for tag_list in df_products['flavor_tags'].str.split(',')
        if isinstance(tag_list, list)
        for tag in tag_list
        if tag.strip()
    )))

    all_flavor_embeddings = get_embeddings(df_products['flavor_tags'].tolist())
    df_products['flavor_tag_embedding'] = all_flavor_embeddings


    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("Let’s find your perfect brew!")

        st.markdown("---")
        st.subheader("Step 1: What are you looking for?")
        drink_type = st.radio("Choose a category:", ["Coffee", "Tea", "Other"], key="drink_type_radio")

        st.markdown("---")
        st.subheader("Step 2: How do you brew?")
        
        brew_grind_options_user = [] # This will hold the user's selected grind/brew types
        
        uses_keurig = st.checkbox("Do you use a Keurig (for pods)?", key="uses_keurig_checkbox")
        if uses_keurig:
            brew_grind_options_user.append("Pods") 
        
        # Default options for multiselect based on available data for the selected drink type
        default_other_grind = []
        if drink_type == "Coffee":
            coffee_products = df_products[df_products['category'].str.contains('Coffee', case=False, na=False)]
            # Check for existing grind types in coffee products
            available_grind_types = coffee_products['grind_type'].str.lower().unique()
            if 'ground' in available_grind_types:
                default_other_grind.append("Ground")
            if 'whole bean' in available_grind_types:
                default_other_grind.append("Whole Bean")
            
            # If no explicit defaults and no keurig, default to ground
            if not default_other_grind and not uses_keurig:
                default_other_grind = ["Ground"]
        
        selected_other_grind_types = st.multiselect(
            "Select other grind type(s):",
            ["Ground", "Whole Bean"],
            default=default_other_grind,
            key="other_grind_type_multiselect"
        )
        brew_grind_options_user.extend(selected_other_grind_types)
        brew_grind_options_user = list(set(brew_grind_options_user)) # Ensure unique elements


        st.markdown("---")
        st.subheader("Step 3: Tell us about your ideal flavor.")

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
                    ai_suggested_tags_filtered = [
                        t.strip() for t in ai_tags_raw.split(',') if t.strip() in all_flavor_suggestions
                    ]

                    if ai_suggested_tags_filtered:
                        st.session_state.ai_suggested_tags = ai_tags_filtered
                        st.markdown(f"**We think you might like these flavors:** {', '.join(ai_suggested_tags_filtered)}")
                        agree_to_ai_tags = st.radio(
                            "Are these on point?",
                            ["Yes, use these", "No, I'll pick"],
                            key="ai_tags_agreement"
                        )
                        if agree_to_ai_tags == "No, I'll pick":
                            st.session_state.ai_suggested_tags = []
                    else:
                        st.warning("We had trouble understanding your flavor style. Please try typing again or pick manually below.")
                        st.session_session.ai_suggested_tags = []
                except Exception as e:
                    st.warning(f"AI flavor suggestion failed: {e}. Please try typing again or pick manually below.")
                    st.session_state.ai_suggested_tags = []

        flavor_input = st.multiselect(
            "Or, select specific flavor notes:",
            options=all_flavor_suggestions,
            default=st.session_state.ai_suggested_tags,
            key="manual_flavor_select"
        )

        if not flavor_input and not description_input.strip():
            if st.checkbox("I'm not sure about flavors, show me popular types!", key="not_sure_flavors"):
                st.info("Okay, we'll suggest some popular general categories for you!")
                st.session_state.explore_categories = True
            else:
                st.session_state.explore_categories = False
        else:
            st.session_state.explore_categories = False

        st.markdown("---")
        st.subheader("Step 4: Roast Preference (for Coffee)")
        roast_preference = "No preference"
        if drink_type == "Coffee":
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
            filtered_products = df_products.copy()

            # 1. Filter by Drink Type (Category) - Strict
            if drink_type and drink_type != "Other":
                initial_filter_len = len(filtered_products)
                category_mask = filtered_products['category'].str.contains(drink_type, case=False, na=False)
                filtered_products = filtered_products[category_mask]
                
                if filtered_products.empty:
                    st.warning(f"No products found specifically for '{drink_type}' matching previous filters. Broadening search to all categories for remaining filters.")
                    filtered_products = df_products.copy() # Revert if filter made it empty

            # 2. Filter by Grind Type (using the new 'grind_type' column) - Strict
            if brew_grind_options_user:
                initial_filter_len = len(filtered_products)
                grind_type_mask = pd.Series([False] * len(filtered_products), index=filtered_products.index)
                
                for option in brew_grind_options_user:
                    if option == "Pods":
                        # Check grind_type column AND category for 'Pods' for robustness
                        grind_type_mask = grind_type_mask | \
                                          (filtered_products['grind_type'].str.contains("pod", case=False, na=False)) | \
                                          (filtered_products['category'].str.contains("pod", case=False, na=False))
                    else:
                        grind_type_mask = grind_type_mask | \
                                          (filtered_products['grind_type'].str.contains(option, case=False, na=False))
                
                filtered_products = filtered_products[grind_type_mask]

                if filtered_products.empty:
                    st.warning(f"No products found matching your selected brew/grind type(s): {', '.join(brew_grind_options_user)}. Ignoring grind type filter to show other relevant products.")
                    filtered_products = df_products.copy() # Revert to initial products if this filter makes it empty

            else: # If no grind types are selected, assume non-pod general products
                st.info("No brew method selected. Automatically excluding 'Pods' and showing all 'Ground'/'Whole Bean' products.")
                filtered_products = filtered_products[
                    ~filtered_products['grind_type'].str.contains("pod", case=False, na=False) &
                    ~filtered_products['category'].str.contains("pod", case=False, na=False)
                ]
                if filtered_products.empty:
                    st.warning("No non-pod products found. Showing all products.")
                    filtered_products = df_products.copy()


            # 3. Filter by Roast Preference (only for Coffee) - Strict
            if drink_type == "Coffee" and roast_preference != "No preference":
                initial_filter_len = len(filtered_products)
                roast_mask = filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
                filtered_products = filtered_products[roast_mask]
                if filtered_products.empty:
                    st.warning(f"No coffee found with '{roast_preference}' roast level matching other criteria. Ignoring roast level filter.")
                    filtered_products = df_products.copy() # Revert to initial products if this filter makes it empty


            # Now, apply flavor matching or surprise logic
            recommendations = pd.DataFrame()

            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'. Showing top general favorites from all products.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42)

            elif flavor_input:
                products_for_similarity = filtered_products.copy()

                products_for_similarity['flavor_overlap_score'] = products_for_similarity['flavor_tags'].apply(
                    lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else []))
                )

                direct_matches = products_for_similarity[products_for_similarity['flavor_overlap_score'] > 0].sort_values(
                    by='flavor_overlap_score', ascending=False
                )

                if not direct_matches.empty:
                    recommendations = direct_matches.head(5)
                else:
                    st.info("No direct flavor matches found. Searching for similar flavor profiles using AI...")
                    try:
                        user_input_embedding = embedder.encode(", ".join(flavor_input), convert_to_tensor=True)
                        
                        valid_embeddings_df = products_for_similarity[
                            products_for_similarity['flavor_tag_embedding'].apply(lambda x: isinstance(x, torch.Tensor) and x.numel() > 0 and not torch.equal(x, torch.zeros_like(x)))
                        ].copy()

                        if not valid_embeddings_df.empty:
                            product_embeddings_list = [emb.cpu() for emb in valid_embeddings_df['flavor_tag_embedding'].tolist()]
                            product_embeddings_tensor = util.cat_embeddings_to_tensor(product_embeddings_list)
                            
                            cosine_scores = util.cos_sim(user_input_embedding.cpu(), product_embeddings_tensor)[0]
                            valid_embeddings_df['similarity_score'] = cosine_scores.cpu().numpy()

                            recommendations = valid_embeddings_df.sort_values(by='similarity_score', ascending=False).head(5)
                            
                            if not recommendations.empty:
                                st.success("Found some great matches with similar flavor profiles!")
                            else:
                                st.warning("Even with AI, we couldn't find close flavor matches after applying all filters. Showing some popular options instead.")
                                recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                        else:
                            st.warning("No products with valid flavor embeddings to compare after applying filters. Showing popular options.")
                            recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)


                    except Exception as e:
                        st.error(f"Error during semantic similarity search: {e}. Showing popular options.")
                        recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)

            elif st.session_state.explore_categories and not surprise_me:
                st.markdown("### Explore by Flavor Category:")
                def categorize_flavor_simple(row):
                    flavors = str(row['specific_flavors']).lower()
                    if 'chocolate' in flavors or 'caramel' in flavors or 'vanilla' in flavors or 'sweet' in flavors:
                        return 'Sweet & Dessert-like'
                    elif 'nutty' in flavors or 'pecan' in flavors or 'almond' in flavors or 'hazelnut' in flavors:
                        return 'Nutty & Rich'
                    elif 'berry' in flavors or 'citrus' in flavors or 'fruity' in flavors or 'floral' in flavors:
                        return 'Fruity & Floral'
                    elif 'dark chocolate' in flavors or 'smoky' in flavors or 'bold' in str(row['short_description']).lower() or 'intense' in str(row['short_description']).lower():
                        return 'Bold & Intense'
                    elif 'smooth' in flavors or 'creamy' in flavors or 'mild' in str(row['short_description']).lower() or 'balanced' in str(row['short_description']).lower():
                        return 'Smooth & Balanced'
                    return 'General Favorites'

                if 'flavor_category' not in filtered_products.columns or filtered_products['flavor_category'].isnull().all():
                     filtered_products['flavor_category'] = filtered_products.apply(categorize_flavor_simple, axis=1)

                top_categories = filtered_products['flavor_category'].value_counts().head(5).index.tolist()
                if top_categories:
                    chosen_category = st.selectbox("Select a popular flavor category:", options=top_categories, key="chosen_category_select")
                    recommendations = filtered_products[filtered_products['flavor_category'] == chosen_category].head(5)
                    st.info(f"Showing popular products in the '{chosen_category}' category.")
                else:
                    st.warning("No categories found after filtering. Showing general popular options.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42)

            else:
                st.info("Please describe your ideal coffee, select flavor notes, or check 'Surprise Me!' to get recommendations.")
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42)


        # --- Display Recommendations ---
        if not recommendations.empty:
            st.markdown("### Your Top Matches:")
            for idx, row in recommendations.iterrows():
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if row['image_url'] and isinstance(row['image_url'], str) and row['image_url'].startswith("http"):
                        try:
                            st.image(row['image_url'], use_container_width=True)
                        except Exception:
                            st.caption("Image not available.")
                    else:
                        st.caption("No image available.")
                with col2:
                    st.subheader(row['name'])
                    st.markdown(f"**Category:** {row['category']}")
                    # Use grind_type instead of brew_method for display
                    if row['grind_type'].strip():
                        st.markdown(f"**Grind Type:** {row['grind_type']}")
                    if row['roast_level'].strip():
                        st.markdown(f"**Roast Level:** {row['roast_level']}")
                    
                    clean_short_description = str(row['short_description']).strip()
                    if clean_short_description:
                        st.write(f"*{clean_short_description}*")
                    else:
                        st.write("*No short description available.*")

                    st.markdown(f"**Price:** ${row['price']:.2f}")
                    if row['specific_flavors']:
                        st.markdown(f"**Flavor Notes:** {row['specific_flavors']}")
                    st.markdown(f"[**View Product on Butler Coffee Lab**]({row['bcl_website_link']})")
        else:
            st.error("Apologies! We couldn't find any products that perfectly match all your selections. Please try adjusting your preferences, selecting fewer filters, or letting us 'Surprise Me!'")

else:
    st.error("There was a problem loading the product data. Please check the app configuration and try again later.")
    st.info("Make sure your Google Sheet is published to the web and the service account has viewer access.")
