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

# --- NO CUSTOM CSS APPLIED ---


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

        # Ensure 'grind' is in required columns
        required_cols = ['name', 'category', 'short_description', 'long_description', 'price', 'bcl_website_link', 'grind']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers (e.g., 'Category' becomes 'category', 'Grind' becomes 'grind').")
                return pd.DataFrame()

        # Ensure essential columns exist, add if missing
        if 'brew_method' not in df.columns: # Keeping for compatibility, but 'grind' is preferred
            df['brew_method'] = ''
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''
        if 'image_url' not in df.columns:
            df['image_url'] = ''

        # Convert relevant columns to string type to prevent errors during processing
        for col in ['name', 'category', 'short_description', 'long_description', 'brew_method', 'roast_level', 'bcl_website_link', 'image_url', 'grind', 'price']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else '')
            else:
                df[col] = ''


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
            tags.append("")
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
        
        brew_grind_options_user = [] 
        
        uses_keurig = st.checkbox("Do you use a Keurig (for pods)?", key="uses_keurig_checkbox")
        if uses_keurig:
            brew_grind_options_user.append("Pods") 
        
        default_other_grind = []
        if drink_type == "Coffee":
            coffee_products = df_products[df_products['category'].str.contains('Coffee', case=False, na=False)]
            # Normalize grind types from data for comparison with user selections
            available_grind_types_normalized = coffee_products['grind'].str.strip().str.lower().unique()
            if 'ground' in available_grind_types_normalized:
                default_other_grind.append("Ground")
            if 'whole bean' in available_grind_types_normalized:
                default_other_grind.append("Whole Bean")
            
            if not default_other_grind and not uses_keurig:
                default_other_grind = ["Ground"]
        
        selected_other_grind_types = st.multiselect(
            "Select other grind type(s):",
            ["Ground", "Whole Bean"],
            default=default_other_grind,
            key="other_grind_type_multiselect"
        )
        brew_grind_options_user.extend(selected_other_grind_types)
        # Normalize user selections to lowercase for consistent filtering later
        brew_grind_options_user = [opt.lower() for opt in list(set(brew_grind_options_user))]


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
                        st.session_state.ai_suggested_tags = []
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
            current_filtered_products = df_products.copy()

            st.markdown("---")
            st.markdown("### Debugging Info (For Developers Only):")
            
            # Highlight the Grind Filter issue clearly
            # This logic remains as it helps diagnose data issues
            if 'grind' in current_filtered_products.columns and \
               'pods' in current_filtered_products['grind'].str.strip().str.lower().unique().tolist() and \
               any(g_opt in ['ground', 'whole bean'] for g_opt in brew_grind_options_user):
                st.error("""
                **ATTENTION: GRIND FILTER ISSUE DETECTED!**
                
                The debugging info shows that **after the initial category filter, your dataset predominantly contains 'Pods' coffee, but you selected 'Ground' or 'Whole Bean' grind type.**
                
                **This indicates that 'Ground' or 'Whole Bean' coffee products in your Google Sheet might be categorized differently (e.g., as 'Bags' instead of 'Coffee').**
                
                The filter logic has been adjusted to account for 'Bags' when 'Coffee' is selected with 'Ground' or 'Whole Bean' grinds. If you still don't see results, please confirm your Google Sheet's 'Category' and 'Grind' columns are accurate for your coffee products.
                """)

            st.write(f"**Initial DataFrame shape:** {current_filtered_products.shape}")
            st.write(f"**Unique 'Grind' values (from raw data):** {current_filtered_products['grind'].unique().tolist()}")
            st.markdown("---")


            # 1. Filter by Drink Type (Category) - MODIFIED TO INCLUDE "BAGS" FOR COFFEE GRINDS
            if drink_type and drink_type != "Other":
                category_mask = current_filtered_products['category'].str.contains(drink_type, case=False, na=False)
                
                # If "Coffee" is selected and user wants Ground/Whole Bean, also include "Bags" category
                if drink_type == "Coffee" and any(g_opt in ['ground', 'whole bean'] for g_opt in brew_grind_options_user):
                    bags_mask = current_filtered_products['category'].str.contains("Bags", case=False, na=False)
                    category_mask = category_mask | bags_mask
                
                if category_mask.any():
                    current_filtered_products = current_filtered_products[category_mask]
                else:
                    st.warning(f"No products found specifically for '{drink_type}' (and potentially 'Bags' for coffee). Proceeding with all categories to apply other filters.")
                    current_filtered_products = df_products.copy() # Revert to full dataset if no matches for category
            
            st.write(f"**DataFrame shape after Category Filter:** {current_filtered_products.shape}")
            # Show normalized unique grind values at this point
            st.write(f"**Unique 'Grind' values (after Category Filter, normalized):** {current_filtered_products['grind'].str.strip().str.lower().unique().tolist()}")


            # --- REWRITTEN GRIND FILTERING (Same as last version, as it correctly handles user input vs data) ---
            if brew_grind_options_user:
                # Normalize the 'grind' column in the DataFrame for consistent matching
                normalized_df_grind = current_filtered_products['grind'].str.strip().str.lower()
                
                # Create a boolean mask for products matching any of the user's selected grind types
                grind_match_mask = pd.Series([False] * len(current_filtered_products), index=current_filtered_products.index)
                
                st.write(f"**User selected grind options (normalized):** {brew_grind_options_user}")
                
                for user_option in brew_grind_options_user:
                    # Check if the normalized DataFrame grind value *contains* the normalized user option
                    # This handles cases like "Ground (for Drip)" matching "ground"
                    grind_match_mask = grind_match_mask | normalized_df_grind.str.contains(user_option, na=False)
                
                temp_df_after_grind = current_filtered_products[grind_match_mask]
                
                if not temp_df_after_grind.empty:
                    current_filtered_products = temp_df_after_grind
                    st.write(f"**DataFrame shape after Grind Filter (matched):** {current_filtered_products.shape}")
                    st.write(f"**'Grind' values of matched products (sample):** {current_filtered_products['grind'].head(5).tolist()}")
                else:
                    st.warning(f"No products found matching your selected brew/grind type(s): {', '.join([opt.capitalize() for opt in brew_grind_options_user])}. Adjusting recommendations based on other preferences.")
                    st.write(f"**DataFrame shape after Grind Filter (no match, reverted):** {current_filtered_products.shape}")
                    st.write(f"**Original 'Grind' values before this filter (sample):** {current_filtered_products['grind'].head(5).tolist()}")
                    # No change to current_filtered_products if no match, effectively ignoring the filter for this step
                    pass 
            else: # If no grind types are selected by the user
                st.info("No brew method selected. Automatically excluding 'Pods' and showing all 'Ground'/'Whole Bean' products.")
                normalized_df_grind = current_filtered_products['grind'].str.strip().str.lower()
                # Exclude anything containing "pod" if no specific grind is selected
                non_pod_mask = ~normalized_df_grind.str.contains("pod", na=False)
                temp_df_after_non_pod_filter = current_filtered_products[non_pod_mask]
                if not temp_df_after_non_pod_filter.empty:
                    current_filtered_products = temp_df_after_non_pod_filter
                    st.write(f"**DataFrame shape after Grind Filter (default non-pod):** {current_filtered_products.shape}")
                    st.write(f"**'Grind' values of default non-pod products (sample):** {current_filtered_products['grind'].head(5).tolist()}")
                else:
                    st.warning("No non-pod products found based on other filters. Showing all products regardless of grind type.")
                    st.write(f"**DataFrame shape after Grind Filter (default non-pod, no match, reverted):** {current_filtered_products.shape}")
                    st.write(f"**Original 'Grind' values before this filter (sample):** {current_filtered_products['grind'].head(5).tolist()}")
                    pass


            # 3. Filter by Roast Preference (only for Coffee)
            if drink_type == "Coffee" and roast_preference != "No preference":
                roast_mask = current_filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
                temp_df_after_roast = current_filtered_products[roast_mask]
                if not temp_df_after_roast.empty:
                    current_filtered_products = temp_df_after_roast
                else:
                    st.warning(f"No coffee found with '{roast_preference}' roast level matching other criteria. Ignoring roast level filter.")
                    pass
            
            st.write(f"**DataFrame shape after Roast Filter:** {current_filtered_products.shape}")
            st.markdown("---")


            # Now, apply flavor matching or surprise logic
            recommendations = pd.DataFrame()

            if surprise_me:
                if not current_filtered_products.empty:
                    recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'. Showing top general favorites from all products.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42)

            elif flavor_input:
                products_for_similarity = current_filtered_products.copy()

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
                                recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)
                        else:
                            st.warning("No products with valid flavor embeddings to compare after applying filters. Showing popular options.")
                            recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)


                    except Exception as e:
                        st.error(f"Error during semantic similarity search: {e}. Showing popular options.")
                        recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)

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

                if 'flavor_category' not in current_filtered_products.columns or current_filtered_products['flavor_category'].isnull().all():
                     current_filtered_products['flavor_category'] = current_filtered_products.apply(categorize_flavor_simple, axis=1)

                top_categories = current_filtered_products['flavor_category'].value_counts().head(5).index.tolist()
                if top_categories:
                    chosen_category = st.selectbox("Select a popular flavor category:", options=top_categories, key="chosen_category_select")
                    recommendations = current_filtered_products[current_filtered_products['flavor_category'] == chosen_category].head(5)
                    st.info(f"Showing popular products in the '{chosen_category}' category.")
                else:
                    st.warning("No categories found after filtering. Showing general popular options.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42)

            else:
                st.info("Please describe your ideal coffee, select flavor notes, or check 'Surprise Me!' to get recommendations.")
                if not current_filtered_products.empty:
                    recommendations = current_filtered_products.sample(min(5, len(current_filtered_products)), random_state=42)
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
                    if row['grind'].strip():
                        st.markdown(f"**Grind Type:** {row['grind']}")
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
