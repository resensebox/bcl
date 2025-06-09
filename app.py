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

# --- Streamlit Page Configuration (MUST BE THE FIRST ST. COMMAND) ---
# This must be the very first Streamlit command executed in your script.
st.set_page_config(
    page_title="Butler Coffee Lab – Flavor Match App",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# --- Configuration ---
# Replace with your actual Google Sheet URL
GOOGLE_SHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE"
# This file is primarily for local testing if not using Streamlit Cloud secrets
SERVICE_ACCOUNT_FILE = "service_account.json"

# --- Initialize OpenAI Client ---
try:
    # Access OpenAI API key from Streamlit secrets
    openai_api_key = st.secrets["open_api_key"] # Use the key name as defined in your secrets.toml
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("OpenAI API key not found. Please add 'open_api_key' to your Streamlit secrets.")
    st.stop() # Stop the app if API key is missing

# --- Initialize Sentence Transformer Model ---
# This model will be used for semantic similarity for flavor matching.
# It's cached to load only once across reruns.
@st.cache_resource
def load_embedding_model():
    # 'all-MiniLM-L6-v2' is a good balance of speed and performance.
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to reduce API calls
def load_data_from_google_sheets():
    """Loads product data from the Google Sheet."""
    try:
        # Authenticate with Google Sheets using Streamlit secrets
        # The key for service account info should match what's in secrets.toml
        if "google_service_account" in st.secrets:
            gc = gspread.service_account_from_dict(st.secrets["google_service_account"])
        else:
            # Fallback for local testing if secrets.toml isn't used for service account
            gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        
        sh = gc.open_by_url(GOOGLE_SHEET_URL)
        worksheet = sh.worksheet("Sheet1") # Adjust sheet name if different (e.g., "Products")
        
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Convert column names to a consistent format (lowercase, no spaces/special chars)
        # Use regex to clean column names for easier access (e.g., 'Category / Type' -> 'category_type')
        df.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns]
        
        # Ensure essential columns exist after renaming
        required_cols = ['name', 'category_type', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers and ensure they conform to expected naming after cleaning (e.g., 'Category / Type' becomes 'category_type').")
                return pd.DataFrame() # Return empty DataFrame on error
        
        # Initialize placeholder columns if they don't exist in the sheet
        if 'brew_method' not in df.columns:
            df['brew_method'] = ''
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''

        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.info("Please ensure your Google Sheet URL is correct, the `service_account.json` (or Streamlit secrets) is properly configured, and the sheet is shared with the service account email.")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24) # Cache images for a day
def scrape_image(url):
    """Scrapes the main product image from a given URL."""
    if not url or url == '#':
        return None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Prioritize Open Graph meta tag, then common img selectors
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image.get('content')

        # Common selectors for product images. Inspect BCL's site for best results.
        img_tag = soup.find('img', class_=lambda x: x and ('product-image' in x or 'main-image' in x))
        if not img_tag:
            img_tag = soup.find('img', alt=lambda x: x and ('product' in x.lower())) # Generic alt text search
        
        if img_tag:
            src = img_tag.get('src') or img_tag.get('data-src') # Check both src and data-src
            if src:
                # Ensure the URL is absolute
                if not src.startswith(('http', 'https')):
                    # Attempt to construct absolute URL if relative
                    base_url = '/'.join(url.split('/')[:3]) # e.g., https://example.com
                    src = f"{base_url}{src}" if src.startswith('/') else f"{base_url}/{src}"
                return src
        
        st.warning(f"Could not find a suitable image for {url}. Please check the URL or provide a more specific selector.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping image from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping image from {url}: {e}")
        return None

@st.cache_data(ttl=60*60*24) # Cache embeddings for a day
def get_embeddings(texts):
    """Generates embeddings for a list of texts using the pre-loaded model."""
    # Ensure texts are valid strings before encoding
    valid_texts = [str(t) for t in texts if pd.notna(t) and t != '']
    if not valid_texts:
        return [np.array([])] # Return empty array for empty input
    return embedder.encode(valid_texts, convert_to_tensor=True)

@st.cache_data(ttl=60*60*24) # Cache LLM calls for a day to save tokens/time
def generate_llm_summary(text, product_name):
    """Generates a concise, playful summary for a product using an LLM."""
    if not client:
        return "AI summary unavailable."
    if not text:
        return f"A delightful {product_name}." # Default if no description

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Consider 'gpt-4o' or 'gpt-4' for higher quality, if budget allows
            messages=[
                {"role": "system", "content": "You are a concise and engaging marketing assistant for a coffee/tea lab. Summarize product descriptions into 1-2 playful and enticing sentences for a product card. Focus on flavor notes, aroma, and the overall feel."},
                {"role": "user", "content": f"Summarize the following product description for '{product_name}':\n\n{text}"}
            ],
            max_tokens=60, # Keep it short and punchy
            temperature=0.7 # A bit creative, but stays on topic
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Could not generate AI summary for '{product_name}': {e}")
        return f"A delightful {product_name}." # Fallback text

@st.cache_data(ttl=60*60*24) # Cache LLM calls for a day
def generate_flavor_profile_summary(flavor_input, drink_type, recommendations_names):
    """Generates a playful 3-sentence flavor profile summary based on user input and recommendations."""
    if not client:
        return "AI flavor profile summary unavailable."
    
    product_list_str = ", ".join(recommendations_names) if recommendations_names else "your selected type of product"
    prompt_flavor = flavor_input if flavor_input else "general flavor preferences"

    prompt = f"""Based on the user's preferences for {drink_type} with flavor notes like '{prompt_flavor}', and considering the recommended products: {product_list_str}.

    Write a short, playful, 3-sentence summary of their flavor profile.
    Sentence 1: Describe their taste preference (e.g., "You love mellow mornings with notes of chocolate and caramel.").
    Sentence 2: Describe the general vibe of the recommended lineup (e.g., "This lineup is cozy, smooth, and just a little nutty—like your perfect Sunday.").
    Sentence 3: Include the mission-driven message: "Each sip supports inclusive employment. Donate here!"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Consider 'gpt-4o' or 'gpt-4' for higher quality
            messages=[
                {"role": "system", "content": "You are a creative and brand-aligned copywriter for Butler Coffee Lab. Your summaries are playful, inviting, and mission-driven. Keep it exactly 3 sentences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150, # Sufficient for 3 sentences
            temperature=0.8 # More creative for the summary
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Could not generate AI flavor profile summary: {e}")
        return """
        You love delightful mornings with unique and engaging flavors.
        This lineup is brewed just for you, designed to elevate your daily ritual.
        Each sip supports inclusive employment. Donate here!
        """

# --- Custom CSS for Brand Styling ---
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
        cursor: pointer;
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
        flex-wrap: wrap; /* Allow wrapping on smaller screens */
    }
    .product-card img {
        border-radius: 8px;
        margin-right: 15px;
        width: 100px; /* Fixed width */
        height: 100px; /* Fixed height */
        object-fit: cover; /* Ensures image covers area without distortion */
        flex-shrink: 0; /* Prevent shrinking on small screens */
    }
    .product-card-details {
        flex-grow: 1;
        min-width: 150px; /* Ensure details don't get too squished */
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
        margin-left: auto; /* Push button to the right */
        flex-shrink: 0; /* Prevent shrinking */
    }
    .product-card .buy-button a {
        background-color: #B35F3A;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 14px;
        display: inline-block; /* Ensure padding applies */
    }
    .product-card .buy-button a:hover {
        background-color: #C9724C;
    }
    /* Responsive adjustments */
    @media (max-width: 600px) {
        .product-card {
            flex-direction: column;
            align-items: flex-start;
        }
        .product-card img {
            margin-right: 0;
            margin-bottom: 10px;
        }
        .product-card-details {
            width: 100%;
        }
        .product-card .buy-button {
            margin-left: 0;
            margin-top: 10px;
            width: 100%;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("☕️ Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

# Load data
df_products = load_data_from_google_sheets()

if not df_products.empty:
    # Generate embeddings for product descriptions
    # Filter out empty or NaN descriptions before generating embeddings
    valid_descriptions = df_products['long_description'].fillna('').tolist()
    # Replace empty strings with a placeholder to prevent issues with embedding model
    cleaned_descriptions = [desc if desc.strip() != '' else 'general product description' for desc in valid_descriptions]

    all_embeddings = get_embeddings(cleaned_descriptions)
    
    # Assign embeddings back to the DataFrame. Handle cases where embedding might be empty.
    df_products['long_description_embedding'] = [
        all_embeddings[i] if all_embeddings[i].size > 0 else np.array([])
        for i in range(len(all_embeddings))
    ]
    
    st.sidebar.header("Tell us your preferences!")

    with st.sidebar.form("flavor_form"):
        st.markdown("**1. What are you in the mood for?**")
        drink_type = st.radio("Drink Type", ["Coffee", "Tea"], horizontal=True, index=0)

        st.markdown("**2. How do you brew?**")
        brew_method = st.multiselect(
            "Brew Method",
            ["Pods", "Ground", "Whole Bean"],
            default=["Ground"] # Default to Ground for convenience
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
        with st.spinner("Finding your perfect match... This may take a moment."):
            # --- Filtering based on categorical inputs ---
            filtered_products = df_products[
                (df_products['category_type'].str.contains(drink_type, case=False, na=False))
            ].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Filter by brew method
            if brew_method:
                filtered_products = filtered_products[
                    filtered_products['brew_method'].fillna('').apply(
                        lambda x: any(b.lower() in x.lower() for b in brew_method)
                    )
                ]

            # Filter by roast preference for coffee
            if drink_type == "Coffee" and roast_preference != "No preference":
                filtered_products = filtered_products[
                    filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
                ]
            
            recommendations = pd.DataFrame()
            
            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42) # Consistent random picks
            elif flavor_input:
                # Semantic similarity search
                # Ensure there are products with valid embeddings to compare against
                products_for_similarity = filtered_products[
                    filtered_products['long_description_embedding'].apply(lambda x: x.size > 0)
                ]

                if not products_for_similarity.empty:
                    # Get embedding for user's flavor input
                    user_embedding = get_embeddings([flavor_input])[0]
                    
                    # Convert list of tensors to a single tensor for batch similarity calculation
                    product_embeddings_tensor = util.cat_embeddings_to_tensor(
                        [e for e in products_for_similarity['long_description_embedding']]
                    )
                    
                    # Calculate cosine similarity
                    cosine_scores = util.cos_sim(user_embedding, product_embeddings_tensor)[0]
                    
                    # Add scores to DataFrame and sort
                    products_for_similarity = products_for_similarity.copy() # Avoid SettingWithCopyWarning
                    products_for_similarity['similarity_score'] = cosine_scores.cpu().numpy()
                    recommendations = products_for_similarity.sort_values(by='similarity_score', ascending=False).head(5)
                else:
                    st.warning("No products with valid descriptions available for semantic matching. Showing some general recommendations.")
                    recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()
            else:
                # If no flavor input and not surprise me, show some general picks
                recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()

        # --- Display Results ---
        if not recommendations.empty:
            st.markdown("---")
            st.markdown('<p class="big-font">Your Personalized Butler Coffee Lab Picks:</p>', unsafe_allow_html=True)

            # AI-generated Flavor Profile Summary
            recommended_names = recommendations['name'].tolist()
            profile_summary = generate_flavor_profile_summary(flavor_input, drink_type, recommended_names)
            st.markdown(f'<p style="font-size:16px; color:#555555;">{profile_summary}</p>', unsafe_allow_html=True)

            for _, product in recommendations.iterrows():
                product_name = product.get('name', 'N/A')
                long_desc = product.get('long_description', '')
                price = product.get('price', 'N/A')
                bcl_link = product.get('bcl_website_link', '#')
                
                # AI-generated short description for the product card
                ai_short_desc = generate_llm_summary(long_desc, product_name)
                
                # You might want to extract top keywords from the long_description
                # or just use the AI summary for conciseness here.
                # For simplicity, let's show the similarity score if available.
                flavor_tags_display = ""
                if 'similarity_score' in product:
                    flavor_tags_display = f"Similarity Score: {product['similarity_score']:.2f}"
                # You could also add specific flavor keywords from your sheet if available
                # e.g., product.get('flavor_keywords', '')

                image_url = scrape_image(bcl_link)
                
                st.markdown(f"""
                <div class="product-card">
                    {"<img src='" + image_url + "' alt='" + product_name + "' />" if image_url else ""}
                    <div class="product-card-details">
                        <h4>{product_name}</h4>
                        <p>{ai_short_desc}</p>
                        <p class="price">${price}</p>
                        <p class="tags">{flavor_tags_display}</p>
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
    st.error("Could not load product data. Please check your Google Sheet URL, service account setup, and column names.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for Butler Coffee Lab.")
