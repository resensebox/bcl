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
# Use the full Google Sheet URL here, or reconstruct it from the ID.
# Example: "https://docs.google.com/spreadsheets/d/1VBnG4kfGOUN3iVH1n14qOUnzivhiU_SsOclCAcWkFI8/edit?usp=sharing"
# The ID you provided is '1VBnG4kfGOUN3iVH1n14qOUnzivhiU_SsOclCAcWkFI8'
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1VBnG4kfGOUN3iVH1n14qOUnzivhiU_SsOclCAcWkFI8/edit?usp=sharing" #

# Initialize OpenAI Client using Streamlit secrets.
# IMPORTANT: Changed this back to st.secrets["open_ai_key"] to match your previous secrets.toml format.
# If your secrets.toml has a nested structure like [openai] then api_key = "...",
# you would keep st.secrets["openai"]["api_key"]. Let's assume the simpler flat structure for now.
try:
    openai_api_key = st.secrets["open_ai_key"] # Access as a direct key, not nested
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("Looks like your OpenAI API key isn't set up correctly in Streamlit Secrets. Please ensure 'open_ai_key' is present in your .streamlit/secrets.toml.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# --- 3. Helper Functions ---

@st.cache_data(ttl=3600)
def load_data_from_google_sheets():
    """Loads product data from the specified Google Sheet."""
    try:
        if "google_service_account" in st.secrets:
            gc = gspread.service_account_from_dict(st.secrets["google_service_account"])
        else:
            st.error("Google service account secrets not found. Please configure 'google_service_account' in your Streamlit secrets.")
            st.stop()
        
        # Open by URL directly. Make sure the URL includes "/edit?usp=sharing" or similar.
        sh = gc.open_by_url(GOOGLE_SHEET_URL)
        worksheet = sh.worksheet("Sheet1") # Adjust sheet name if different
        
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        df.columns = [re.sub(r'[^a-z0-9_]', '', col.lower().replace(' ', '_')) for col in df.columns]
        
        required_cols = ['name', 'category_type', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers (e.g., 'Category / Type' becomes 'category_type').")
                return pd.DataFrame()
        
        if 'brew_method' not in df.columns:
            df['brew_method'] = ''
        if 'roast_level' not in df.columns:
            df['roast_level'] = ''

        return df
    except Exception as e:
        # Provide a more specific error message if the sheet cannot be opened or data processed.
        st.error(f"Uh oh! Couldn't load data from Google Sheets. Error: {e}")
        st.info("Double-check your Google Sheet URL (including /edit?usp=sharing), and ensure the service account email has Viewer access to the sheet.")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def scrape_image(url):
    """Attempts to scrape the main product image from a given URL."""
    if not url or url == '#':
        return None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image.get('content')

        img_tag = soup.find('img', class_=lambda x: x and ('product-image' in x or 'main-image' in x))
        if not img_tag:
            img_tag = soup.find('img', alt=lambda x: x and ('product' in x.lower() or 'coffee' in x.lower() or 'tea' in x.lower()))
        
        if img_tag:
            src = img_tag.get('src') or img_tag.get('data-src')
            if src:
                if not src.startswith(('http', 'https')):
                    base_url = '/'.join(url.split('/')[:3])
                    src = f"{base_url}{src}" if src.startswith('/') else f"{base_url}/{src}"
                return src
        
        st.warning(f"Couldn't find a clear product image for {url}.")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch image from {url}: {e}")
        return None
    except Exception as e:
        st.warning(f"An unexpected error occurred while scraping image from {url}: {e}")
        return None

@st.cache_data(ttl=60*60*24)
def get_embeddings(texts):
    """Generates embeddings for a list of texts using the pre-loaded model."""
    valid_texts = [str(t) for t in texts if pd.notna(t) and t.strip() != '']
    if not valid_texts:
        return [np.array([])]
    return embedder.encode(valid_texts, convert_to_tensor=True)

@st.cache_data(ttl=60*60*24)
def generate_llm_summary(text, product_name):
    """Generates a concise, playful summary for a product using OpenAI's LLM."""
    if not client or not text.strip():
        return f"A delightful product from Butler Coffee Lab."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise, engaging, and playful marketing assistant for Butler Coffee Lab. Summarize product descriptions into 1-2 enticing sentences for a product card. Focus on key flavor notes, aroma, and the overall experience."},
                {"role": "user", "content": f"Summarize the following product description for '{product_name}':\n\n{text}"}
            ],
            max_tokens=60,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"AI couldn't generate a summary for '{product_name}'. Error: {e}")
        return f"A delightful {product_name} from Butler Coffee Lab."

@st.cache_data(ttl=60*60*24)
def generate_flavor_profile_summary(flavor_input, drink_type, recommendations_names):
    """Generates a playful 3-sentence flavor profile summary using OpenAI's LLM."""
    if not client:
        return """
        You love delightful mornings with unique and engaging flavors.
        This lineup is brewed just for you, designed to elevate your daily ritual.
        Each sip supports inclusive employment. Donate here!
        """
    
    product_list_str = ", ".join(recommendations_names) if recommendations_names else "your selected type of product"
    prompt_flavor = flavor_input if flavor_input else "general flavor preferences"

    prompt = f"""Based on the user's preferences for {drink_type} with flavor notes like '{prompt_flavor}', and considering these recommended products: {product_list_str}.

    Write a short, playful, 3-sentence summary of their flavor profile.
    Sentence 1: Describe their taste preference (e.g., "You love mellow mornings with notes of chocolate and caramel.").
    Sentence 2: Describe the general vibe of the recommended lineup (e.g., "This lineup is cozy, smooth, and just a little nutty—like your perfect Sunday.").
    Sentence 3: Include the mission-driven message: "Each sip supports inclusive employment. Donate here!"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative and brand-aligned copywriter for Butler Coffee Lab. Your summaries are playful, inviting, and mission-driven. Keep it exactly 3 sentences."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"AI couldn't generate flavor profile summary. Error: {e}")
        return """
        You love delightful mornings with unique and engaging flavors.
        This lineup is brewed just for you, designed to elevate your daily ritual.
        Each sip supports inclusive employment. Donate here!
        """

# --- 4. Custom CSS for Brand Styling ---
st.markdown("""
<style>
    /* General body and text color */
    .stApp {
        background-color: #FFFFFF; /* Pure white background */
        color: #333333; /* Dark gray for general text */
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #7A492C; /* Darker brown for headings */
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }

    /* Streamlit's default markdown text (e.g., st.markdown("### Find your perfect brew...") ) */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stText"] {
        color: #333333; /* Ensure standard markdown text is dark */
    }

    /* Buttons */
    .stButton>button {
        background-color: #A9876D; /* Warm button color */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #C0A080; /* Lighter on hover */
    }

    /* Product Cards */
    .product-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        color: #333333; /* Ensure text inside cards is dark */
    }
    .product-card img {
        border-radius: 8px;
        margin-right: 15px;
        width: 100px;
        height: 100px;
        object-fit: cover;
        flex-shrink: 0;
        border: 1px solid #E0E0E0;
    }
    .product-card-details {
        flex-grow: 1;
        min-width: 150px;
    }
    .product-card h4 {
        margin-top: 0;
        margin-bottom: 5px;
        color: #7A492C;
    }
    .product-card p {
        font-size: 14px;
        color: #555555; /* Slightly lighter dark gray for paragraphs */
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
        margin-left: auto;
        flex-shrink: 0;
    }
    .product-card .buy-button a {
        background-color: #B35F3A;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 14px;
        display: inline-block;
        transition: background-color 0.3s ease;
    }
    .product-card .buy-button a:hover {
        background-color: #C9724C;
    }

    /* Sidebar styles */
    .stSidebar {
        background-color: #EEDDCC;
        padding: 20px;
        border-right: 1px solid #D0C0B0;
        color: #333333; /* Dark text for sidebar elements */
    }
    .stSidebar .stRadio div {
        display: flex;
        flex-direction: row;
        justify-content: space-around;
    }

    /* --- Messages (Error, Warning, Info) --- */
    div[data-testid="stAlert"] {
        color: #333333 !important; /* Ensure alert text is dark */
    }
    div[data-testid="stAlert"] a {
        color: #007bff !important; /* Make links in alerts stand out */
    }
    div[data-testid="stAlert"] .st-cd { /* For info messages */
        color: #333333 !important;
    }
    div[data-testid="stAlert"] .st-ce { /* For warning messages */
        color: #333333 !important;
    }
    div[data-testid="stAlert"] .st-cf { /* For error messages */
        color: #333333 !important;
    }

    /* Responsive adjustments for smaller screens */
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
        .stSidebar .stRadio div {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 5. Main App Logic ---

st.title("☕️ Butler Coffee Lab – Flavor Match App")
st.markdown("### Find your perfect brew, support a great cause!")

df_products = load_data_from_google_sheets()

if not df_products.empty:
    valid_descriptions = df_products['long_description'].fillna('').tolist()
    cleaned_descriptions = [desc if desc.strip() != '' else 'general product description' for desc in valid_descriptions]

    all_embeddings = get_embeddings(cleaned_descriptions)
    
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
        with st.spinner("Brewing up your perfect recommendations..."):
            filtered_products = df_products.copy()

            filtered_products = filtered_products[
                (filtered_products['category_type'].str.contains(drink_type, case=False, na=False))
            ]

            if brew_method:
                filtered_products = filtered_products[
                    filtered_products['brew_method'].fillna('').apply(
                        lambda x: any(b.lower() in x.lower() for b in brew_method)
                    )
                ]

            if drink_type == "Coffee" and roast_preference != "No preference":
                filtered_products = filtered_products[
                    filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False)
                ]
            
            recommendations = pd.DataFrame()
            
            if surprise_me:
                if not filtered_products.empty:
                    recommendations = filtered_products.sample(min(5, len(filtered_products)), random_state=42)
                else:
                    st.warning("No products found matching your basic type and brew preferences for 'Surprise Me'.")
            elif flavor_input:
                products_for_similarity = filtered_products[
                    filtered_products['long_description_embedding'].apply(lambda x: x.size > 0)
                ].copy()

                if not products_for_similarity.empty:
                    user_embedding = get_embeddings([flavor_input])[0]
                    
                    product_embeddings_tensor = util.cat_embeddings_to_tensor(
                        [e for e in products_for_similarity['long_description_embedding']]
                    )
                    
                    cosine_scores = util.cos_sim(user_embedding, product_embeddings_tensor)[0]
                    
                    products_for_similarity['similarity_score'] = cosine_scores.cpu().numpy()
                    recommendations = products_for_similarity.sort_values(by='similarity_score', ascending=False).head(5)
                else:
                    st.warning("No products with detailed descriptions found for flavor matching. Showing some general recommendations.")
                    recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()
            else:
                st.info("No flavor notes entered. Showing some popular picks based on your brew preferences.")
                recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()

        # --- Display Results ---
        if not recommendations.empty:
            st.markdown("---")
            st.markdown('<p class="big-font">Your Personalized Butler Coffee Lab Picks:</p>', unsafe_allow_html=True)

            recommended_names = recommendations['name'].tolist()
            profile_summary = generate_flavor_profile_summary(flavor_input, drink_type, recommended_names)
            st.markdown(f'<p style="font-size:16px; color:#555555;">{profile_summary}</p>', unsafe_allow_html=True)

            for _, product in recommendations.iterrows():
                product_name = product.get('name', 'N/A')
                long_desc = product.get('long_description', '')
                price = product.get('price', 'N/A')
                bcl_link = product.get('bcl_website_link', '#')
                
                ai_short_desc = generate_llm_summary(long_desc, product_name)
                
                flavor_tags_display = ""
                if 'similarity_score' in product:
                    flavor_tags_display = f"Flavor Match Score: {product['similarity_score']:.2f}"
                
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
            st.warning("Oops! We couldn't find any products that match your specific preferences. Please try adjusting your choices, or select 'Surprise Me' for a random pick!")

else:
    st.error("There was a problem loading the product data. Please check the app's configuration and try again later.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for Butler Coffee Lab.")
