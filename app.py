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

        # Convert relevant columns to string type to prevent errors during processing
        # Handle potential non-string values gracefully before converting
        for col in ['name', 'category', 'short_description', 'long_description', 'brew_method', 'roast_level', 'bcl_website_link', 'image_url', 'price']:
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
        return []
    
    # Pad or handle empty list for embedder.encode
    if not valid_texts:
        return [torch.zeros(embedder.get_sentence_embedding_dimension())] * len(texts) # Return zero vectors if no valid texts
    
    embeddings = embedder.encode(valid_texts, convert_to_tensor=True)
    
    # Map embeddings back to original list length, putting zero vectors for empty/invalid texts
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
            # st.warning(f"Error extracting tags for item {i}: {e}") # Debugging: uncomment for development
            tags.append("") # Append empty string on error
    return tags

# --- 4. Main App Logic ---
st.title("\u2615\ufe0f Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

if not df_products.empty:
    # Pre-process flavor tags only once, considering both name and long_description
    # Check if specific_flavors column is missing or contains any empty strings
    if 'specific_flavors' not in df_products.columns or \
       df_products['specific_flavors'].isnull().any() or \
       (df_products['specific_flavors'] == '').any(): # Check for actual empty strings
        with st.spinner("Analyzing product flavors with AI (this may take a moment on first run or if data updated)..."):
            df_products['specific_flavors'] = extract_flavor_tags(
                df_products['name'], df_products['long_description']
            )
            # Filter out empty or whitespace-only tags
            df_products['specific_flavors'] = df_products['specific_flavors'].apply(
                lambda x: ', '.join([tag.strip() for tag in x.split(',') if tag.strip()]) if isinstance(x, str) else ''
            )

    df_products['flavor_tags'] = df_products['specific_flavors'] # Use this column for consistency

    # Create a comprehensive list of all unique flavor suggestions
    all_flavor_suggestions = sorted(list(set(
        tag.strip()
        for tag_list in df_products['flavor_tags'].str.split(',')
        if isinstance(tag_list, list)
        for tag in tag_list
        if tag.strip()
    )))

    # Generate embeddings for flavor tags (only for non-empty tags)
    # Ensure every row gets an embedding, even if it's a zero vector
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
        
        brew_method_options_user = [] # This will hold the user's selected brew methods
        
        uses_keurig = st.checkbox("Do you use a Keurig (for pods)?", key="uses_keurig_checkbox")
        if uses_keurig:
            brew_method_options_user.append("Pods") # Automatically add Pods if Keurig is checked
        
        # Determine default for ground/whole bean based on available data for Coffee
        default_other_brew = []
        if drink_type == "Coffee":
            # Check if any coffee products are explicitly 'Ground' or 'Whole Bean'
            coffee_products = df_products[df_products['category'].str.contains('Coffee', case=False, na=False)]
            if 'ground' in coffee_products['brew_method'].str.lower().unique() or \
               any("ground" in name.lower() for name in coffee_products['name'].unique()):
                default_other_brew.append("Ground")
            if 'whole bean' in coffee_products['brew_method'].str.lower().unique() or \
               any("whole bean" in name.lower() for name in coffee_products['name'].unique()):
                default_other_brew.append("Whole Bean")
            # If no explicit default, just set empty to avoid pre-selecting
            if not default_other_brew and not uses_keurig:
                default_other_brew = []
            elif "Ground" not in default_other_brew and "Whole Bean" not in default_other_brew and not uses_keurig:
                 default_other_brew = ["Ground"] # Default to Ground if nothing else, and not pods

        selected_other_brew_methods = st.multiselect(
            "Select other brew method(s):",
            ["Ground", "Whole Bean"],
            default=default_other_brew,
            key="other_brew_method_multiselect"
        )
        brew_method_options_user.extend(selected_other_brew_methods)
        brew_method_options_user = list(set(brew_method_options_user)) # Ensure unique elements


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
                        t.strip() for t in ai_tags_raw.split(',') if t.strip() in all_flavor_suggestions
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
            filtered_products = df_products.copy()
            initial_product_count = len(filtered_products)

            # Apply filters sequentially, checking for empty results at each step
            # This helps in giving more specific warnings.

            # 1. Filter by Drink Type (Category)
            if drink_type and drink_type != "Other":
                category_mask = filtered_products['category'].str.contains(drink_type, case=False, na=False)
                if category_mask.any():
                    filtered_products = filtered_products[category_mask]
                else:
                    st.warning(f"No products found specifically for '{drink_type}' matching previous filters. Broadening search to all categories for remaining filters.")
                    filtered_products = df_products.copy() # Revert to full set if this specific filter causes emptiness

            # 2. Filter by Brew Method
            # Create a combined string for brew method keywords in both columns
            filtered_products['combined_brew_info'] = filtered_products['brew_method'].str.lower() + " " + filtered_products['name'].str.lower()

            brew_method_mask = pd.Series([True] * len(filtered_products), index=filtered_products.index) # Start with all True
            
            # If user selected "Pods", include products identified as pods
            if "Pods" in brew_method_options_user:
                # Products are pods if 'pod' or 'k-cup' or 'single serve' appear in combined brew info
                pod_keywords_regex = r'pod|k-cup|k cup|single serve'
                brew_method_mask = brew_method_mask & \
                                   (filtered_products['combined_brew_info'].str.contains(pod_keywords_regex, regex=True, na=False))
                
                # If other brew methods (Ground, Whole Bean) are also selected, combine them
                other_methods = [m.lower() for m in brew_method_options_user if m != "Pods"]
                if other_methods:
                    other_methods_regex = '|'.join(re.escape(m) for m in other_methods) # Escape for regex
                    brew_method_mask = brew_method_mask | \
                                       (filtered_products['combined_brew_info'].str.contains(other_methods_regex, regex=True, na=False))

            # If user did NOT select "Pods", explicitly exclude them
            elif "Pods" not in brew_method_options_user and brew_method_options_user: # Other methods selected but not Pods
                pod_keywords_regex = r'pod|k-cup|k cup|single serve'
                brew_method_mask = brew_method_mask & \
                                   (~filtered_products['combined_brew_info'].str.contains(pod_keywords_regex, regex=True, na=False))
                
                # Now apply the other selected brew methods (Ground, Whole Bean)
                allowed_methods_lower = [m.lower() for m in brew_method_options_user]
                if allowed_methods_lower:
                    allowed_methods_regex = '|'.join(re.escape(m) for m in allowed_methods_lower)
                    brew_method_mask = brew_method_mask & \
                                       (filtered_products['combined_brew_info'].str.contains(allowed_methods_regex, regex=True, na=False))
                else: # This case should ideally not happen if brew_method_options_user is not empty and no "Pods"
                    pass # Keep the mask as is (excluding pods)

            elif not brew_method_options_user: # No brew methods selected at all, default to non-pods
                pod_keywords_regex = r'pod|k-cup|k cup|single serve'
                brew_method_mask = brew_method_mask & \
                                   (~filtered_products['combined_brew_info'].str.contains(pod_keywords_regex, regex=True, na=False))
                st.info("No brew method selected. Automatically excluding 'Pods' and showing all 'Ground'/'Whole Bean' products.")


            # Apply the brew method mask
            if brew_method_mask.any():
                filtered_products = filtered_products[brew_method_mask]
            else:
                st.warning(f"No products found matching your selected brew method(s): {brew_method_options_user}. Ignoring brew method filter to show other relevant products.")
                # Revert to the state before this filter was applied if it results in empty
                # This requires careful state management. For now, let's keep it less strict
                # and just proceed with what we have, but warn.

            # Drop the temporary column
            filtered_products = filtered_products.drop(columns=['combined_brew_info'], errors='ignore')


            # 3. Filter by Roast Preference (only for Coffee)
            if drink_type == "Coffee" and roast_preference != "No preference":
                roast_mask = filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
                if roast_mask.any():
                    filtered_products = filtered_products[roast_mask]
                else:
                    st.warning(f"No coffee found with '{roast_preference}' roast level matching other criteria. Ignoring roast level filter for coffee.")
                    pass # Keep `filtered_products` as is if this specific filter causes emptiness


            # Now, apply flavor matching or surprise logic
            recommendations = pd.DataFrame()

            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'. Showing top general favorites from all products.")
                    recommendations = df_products.sample(min(5, len(df_products)), random_state=42) # Fallback to general products

            elif flavor_input:
                products_for_similarity = filtered_products.copy()

                # Calculate flavor overlap score (direct matches)
                products_for_similarity['flavor_overlap_score'] = products_for_similarity['flavor_tags'].apply(
                    lambda tags: len(set(flavor_input) & set([t.strip() for t in tags.split(',')] if isinstance(tags, str) else []))
                )

                # Get initial recommendations based on direct overlap
                direct_matches = products_for_similarity[products_for_similarity['flavor_overlap_score'] > 0].sort_values(
                    by='flavor_overlap_score', ascending=False
                )

                if not direct_matches.empty:
                    recommendations = direct_matches.head(5)
                else:
                    st.info("No direct flavor matches found. Searching for similar flavor profiles using AI...")
                    try:
                        user_input_embedding = embedder.encode(", ".join(flavor_input), convert_to_tensor=True)
                        
                        # Filter out products with empty or invalid embeddings for semantic similarity
                        # This part needs to be very robust to avoid the "No products with valid flavor embeddings" error
                        valid_embeddings_df = products_for_similarity[
                            products_for_similarity['flavor_tag_embedding'].apply(lambda x: isinstance(x, torch.Tensor) and x.numel() > 0 and not torch.equal(x, torch.zeros_like(x)))
                        ].copy()

                        if not valid_embeddings_df.empty:
                            product_embeddings_list = [emb.cpu() for emb in valid_embeddings_df['flavor_tag_embedding'].tolist()]
                            product_embeddings_tensor = util.cat_embeddings_to_tensor(product_embeddings_list)
                            
                            # Ensure user_input_embedding is on CPU for comparison
                            cosine_scores = util.cos_sim(user_input_embedding.cpu(), product_embeddings_tensor)[0]
                            valid_embeddings_df['similarity_score'] = cosine_scores.cpu().numpy()

                            recommendations = valid_embeddings_df.sort_values(by='similarity_score', ascending=False).head(5)
                            
                            if not recommendations.empty:
                                st.success("Found some great matches with similar flavor profiles!")
                            else:
                                st.warning("Even with AI, we couldn't find close flavor matches after applying all filters. Showing some popular options instead.")
                                recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42) # Fallback to filtered products for sampling
                        else:
                            st.warning("No products with valid flavor embeddings to compare after applying filters. Showing popular options.")
                            recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)


                    except Exception as e:
                        st.error(f"Error during semantic similarity search: {e}. Showing popular options.")
                        recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)

            # Fallback if no specific selection and "not sure" is checked
            elif st.session_state.explore_categories and not surprise_me:
                st.markdown("### Explore by Flavor Category:")
                # Simple rule-based categorization if AI categorization isn't implemented elsewhere
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

                # Apply this categorization if 'flavor_category' is not already populated
                # or if products changed due to filters
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


            else: # No specific input, no surprise, no "not sure" selected
                st.info("Please describe your ideal coffee, select flavor notes, or check 'Surprise Me!' to get recommendations.")
                # Show general favorites if no input
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    # Fallback to full dataframe if even the initial filtered_products is empty
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
                    
                    # Robust short_description display
                    clean_short_description = str(row['short_description']).strip()
                    if clean_short_description:
                        st.write(f"*{clean_short_description}*")
                    else:
                        st.write("*No short description available.*")

                    st.markdown(f"**Price:** ${row['price']:.2f}") # Format price
                    if row['specific_flavors']:
                        st.markdown(f"**Flavor Notes:** {row['specific_flavors']}")
                    st.markdown(f"[**View Product on Butler Coffee Lab**]({row['bcl_website_link']})")
        else:
            st.error("Apologies! We couldn't find any products that perfectly match all your selections. Please try adjusting your preferences, selecting fewer filters, or letting us 'Surprise Me!'")

else:
    st.error("There was a problem loading the product data. Please check the app configuration and try again later.")
    st.info("Make sure your Google Sheet is published to the web and the service account has viewer access.")
