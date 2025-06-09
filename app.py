# streamlit_app.py

import streamlit as st
import pandas as pd
import gspread
import re
from bs4 import BeautifulSoup
import requests
import json
import random # For surprise me option
from sentence_transformers import SentenceTransformer, util # For embeddings
import openai # For OpenAI API (or similar LLM library)

# --- Configuration ---
# IMPORTANT: Replace with your actual Google Sheet URL
GOOGLE_SHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE"
# For local development, place service_account.json in the same directory.
# For Streamlit Cloud, use st.secrets for both service account and API keys.
SERVICE_ACCOUNT_FILE = "service_account.json"

# --- Initialize AI Models ---
# Load a pre-trained Sentence Transformer model for embeddings
# 'all-MiniLM-L6-v2' is a good balance of size and performance for many tasks.
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# Initialize OpenAI Client if API key is available
openai_client = None
if "OPENAI_API_KEY" in st.secrets:
    openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.warning("OpenAI API key not found in Streamlit secrets. AI summary generation will be disabled.")


# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to reduce API calls
def load_data_from_google_sheets():
    """Loads product data from the Google Sheet."""
    try:
        # For Streamlit Cloud deployment, use st.secrets for the service account info
        if "service_account_info" in st.secrets:
            gc = gspread.service_account_from_dict(st.secrets["service_account_info"])
        else:
            gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        
        sh = gc.open_by_url(GOOGLE_SHEET_URL)
        worksheet = sh.worksheet("Sheet1") # Replace "Sheet1" with your actual sheet name if different
        
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Convert column names to a consistent format (e.g., lowercase, no spaces)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        required_cols = ['name', 'category_/_type', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers.")
                return pd.DataFrame() 
        
        # Ensure 'brew_method' and 'roast_level' columns exist or create placeholders
        if 'brew_method' not in df.columns:
            st.warning("Column 'brew_method' not found. Please add it to your Google Sheet.")
            df['brew_method'] = "" # Add an empty column
        if 'roast_level' not in df.columns:
            st.warning("Column 'roast_level' not found. Please add it to your Google Sheet for coffee products.")
            df['roast_level'] = "" # Add an empty column

        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.info("Please ensure your `service_account.json` is correct (or secrets are set) and the sheet is shared with the service account email.")
        return pd.DataFrame()

# No longer generating static tags. We'll use embeddings of descriptions.
# def generate_tags(description):
#     """Generates a list of tags from product descriptions."""
#     # ... (previous logic, but we're moving to embeddings) ...
#     return list(set(found_tags))

@st.cache_data(ttl=3600*24) # Cache images for a day
def scrape_image(url):
    """Scrapes the main product image from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Prioritize Open Graph meta tag, then common img selectors
        img_tag = soup.find('meta', property='og:image')
        if img_tag and img_tag.get('content'):
            return img_tag.get('content')
        
        img_tag = soup.find('img', class_=lambda x: x and ('product-image' in x or 'main-image' in x or 'wp-post-image' in x))
        if not img_tag:
            img_tag = soup.find('img', src=re.compile(r'(product|thumbnail|image)')) # Broader search

        if img_tag:
            src = img_tag.get('src') or img_tag.get('data-src')
            if src:
                if not src.startswith(('http', 'https')):
                    # Attempt to construct absolute URL
                    from urllib.parse import urljoin
                    src = urljoin(url, src)
                return src
        
        st.warning(f"Could not find a suitable image for {url}. Please check the URL or provide a more specific selector.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping image from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping image from {url}: {e}")
        return None

# --- AI Summary Generation ---
def generate_ai_summary(user_flavor_input, recommended_products_df):
    if not openai_client:
        return "AI summary not available. Please configure your OpenAI API key."

    product_names = recommended_products_df['name'].tolist()
    product_descriptions = recommended_products_df['short_description'].tolist()
    
    prompt = f"""
    You are a playful and mission-driven barista for Butler Coffee Lab.
    Based on the user's input flavor preferences: "{user_flavor_input}"
    And the following recommended products:
    {', '.join([f'{name}: {desc}' for name, desc in zip(product_names, product_descriptions)])}

    Write a short, playful, and inspiring 3-sentence summary that captures the user's taste profile and connects it to Butler Coffee Lab's mission.
    Example: "You love mellow mornings with notes of chocolate and caramel. This lineup is cozy, smooth, and just a little nutty—like your perfect Sunday. Each sip supports inclusive employment. Donate here!"

    Your summary:
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Or "gpt-4" for better quality
            messages=[
                {"role": "system", "content": "You are a helpful, creative, and playful barista for Butler Coffee Lab."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7 # Creativity level
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        st.error(f"OpenAI API error: {e}")
        return "Failed to generate AI summary. Please try again later."
    except Exception as e:
        st.error(f"An unexpected error occurred during AI summary generation: {e}")
        return "Failed to generate AI summary. Please try again later."


# --- Streamlit App ---

st.set_page_config(
    page_title="Butler Coffee Lab – Flavor Match App",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Custom CSS for styling (placeholder for brand vibe)
st.markdown("""
<style>
    .reportview-container {
        background: #FDF7E7; /* Light pastel background */
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #7A492C; /* Darker brown for headings */
    }
    .stButton>button {
        background-color: #A9876D; /* Warm button color */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #C0A080; /* Lighter on hover */
    }
    .product-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
    }
    .product-card img {
        border-radius: 8px;
        margin-right: 15px;
        width: 100px; /* Adjust as needed */
        height: 100px; /* Adjust as needed */
        object-fit: cover;
    }
    .product-card-details {
        flex-grow: 1;
    }
    .product-card h4 {
        color: #7A492C;
        margin-top: 0;
        margin-bottom: 5px;
    }
    .product-card p {
        font-size: 14px;
        color: #555555;
        margin-bottom: 5px;
    }
    .product-card .price {
        font-weight: bold;
        color: #B35F3A;
        font-size: 15px;
    }
    .product-card .tags {
        font-style: italic;
        font-size: 12px;
        color: #888888;
    }
    .product-card .buy-button {
        text-align: right;
    }
    .product-card .buy-button a {
        background-color: #B35F3A;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 14px;
    }
    .product-card .buy-button a:hover {
        background-color: #C9724C;
    }
</style>
""", unsafe_allow_html=True)

st.title("☕️ Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

# Load data
df_products = load_data_from_google_sheets()

if not df_products.empty:
    # Generate embeddings for product descriptions
    # Using 'long_description' for richer semantic information
    # Filter out empty descriptions before encoding
    df_products['embedding'] = df_products['long_description'].apply(
        lambda x: embedder.encode(x, convert_to_tensor=True) if pd.notna(x) and x.strip() != '' else None
    )
    # Remove products without a valid embedding (e.g., if description was empty)
    df_products = df_products.dropna(subset=['embedding']).reset_index(drop=True)

    if df_products.empty:
        st.error("No products available after generating embeddings. Please check your product descriptions.")
        st.stop()


    st.sidebar.header("Tell us your preferences!")

    with st.sidebar.form("flavor_form"):
        st.markdown("**1. What are you in the mood for?**")
        drink_type = st.radio("Drink Type", ["Coffee", "Tea"], horizontal=True, index=0)

        st.markdown("**2. How do you brew?**")
        brew_method = st.multiselect(
            "Brew Method",
            ["Pods", "Ground", "Whole Bean"],
            default=["Ground"]
        )

        st.markdown("**3. What flavors do you love?**")
        flavor_input = st.text_input(
            "Enter flavor notes (e.g., chocolate, nutty, smooth)",
            placeholder="e.g., chocolate, caramel, fruity, bold"
        )

        roast_preference = "No preference"
        if drink_type == "Coffee":
            st.markdown("**4. How do you like your coffee roasted?**")
            roast_preference = st.radio(
                "Roast/Intensity",
                ["Light", "Medium", "Dark", "No preference"],
                horizontal=True,
                index=3
            )
        
        st.markdown("**5. Feeling adventurous?**")
        surprise_me = st.checkbox("Surprise Me!")

        submitted = st.form_submit_button("Find My Perfect Match!")

    if submitted:
        # --- AI-Powered Recommendations Logic ---
        
        # Apply initial filters based on explicit choices
        filtered_products = df_products[
            (df_products['category_/_type'].str.contains(drink_type, case=False, na=False)) &
            (df_products['brew_method'].fillna('').apply(lambda x: any(b.lower() in x.lower() for b in brew_method)))
        ]

        if drink_type == "Coffee" and roast_preference != "No preference":
            filtered_products = filtered_products[
                filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
            ]
        
        if filtered_products.empty:
            st.warning("No products found matching your basic drink type, brew method, and roast preferences.")
            st.stop()

        recommendations = pd.DataFrame()
        if surprise_me:
            recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42) # Fixed seed for reproducibility
        else:
            if flavor_input and flavor_input.strip() != "":
                # Generate embedding for user's flavor input
                user_embedding = embedder.encode(flavor_input, convert_to_tensor=True)
                
                # Calculate cosine similarity with all filtered product embeddings
                similarities = util.cos_sim(user_embedding, filtered_products['embedding'].tolist())
                
                # Add similarities to DataFrame and sort
                filtered_products['similarity'] = similarities[0].cpu().numpy()
                
                # Get top N recommendations based on similarity
                recommendations = filtered_products.sort_values(by='similarity', ascending=False).head(5)
                
                if recommendations.empty:
                    st.warning("No close flavor matches found. Showing general recommendations.")
                    recommendations = filtered_products.sample(min(3, len(filtered_products)), random_state=42)
            else:
                # If no flavor input and not surprise me, show some random general picks
                recommendations = filtered_products.sample(min(3, len(filtered_products)), random_state=42)

        # --- Display Results ---
        if not recommendations.empty:
            st.markdown("---")
            st.markdown('<p class="big-font">Your Personalized Butler Coffee Lab Picks:</p>', unsafe_allow_html=True)

            # Flavor Profile Summary (AI-powered)
            ai_summary_text = generate_ai_summary(flavor_input, recommendations)
            st.markdown(f"""
            <p style="font-size:16px; color:#555555;">
            {ai_summary_text}
            </p>
            """, unsafe_allow_html=True)


            for _, product in recommendations.iterrows():
                product_name = product.get('name', 'N/A')
                # Shortened, rewritten description - you might consider another LLM call here
                # or just use short_description if it's already concise.
                display_description = product.get('short_description', product.get('long_description', 'A delightful product.')[:100] + '...')
                price = product.get('price', 'N/A')
                bcl_link = product.get('bcl_website_link', '#')
                
                # Display relevant "tags" (can be inferred from description or specific keywords)
                # For this version, we're relying on semantic similarity, so explicit tags are less critical for matching
                # but can be displayed if a 'tags' column exists from the sheet or is generated
                # For now, let's just use a simplified representation of the long description if no specific 'tags' column.
                product_tags = product.get('long_description', '')
                if len(product_tags) > 100:
                    product_tags = product_tags[:100] + "..." # Truncate for display

                # Scrape image (this might be slow, consider pre-scraping or using a CDN)
                image_url = scrape_image(bcl_link) if bcl_link and bcl_link != '#' else None
                
                st.markdown(f"""
                <div class="product-card">
                    {"<img src='" + image_url + "' alt='" + product_name + "' />" if image_url else ""}
                    <div class="product-card-details">
                        <h4>{product_name}</h4>
                        <p>{display_description}</p>
                        <p class="price">${price}</p>
                        <p class="tags">Description snippet: {product_tags}</p>
                    </div>
                    <div class="buy-button">
                        <a href="{bcl_link}" target="_blank">Buy Now</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("---")
            st.warning("Sorry, no products matched your specific criteria. Please try adjusting your preferences!")

else:
    st.error("Could not load product data or no products with valid embeddings found. Please check your Google Sheet URL, service account setup, column names, and product descriptions.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for Butler Coffee Lab.")
