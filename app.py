# streamlit_app.py

import streamlit as st
import pandas as pd
import gspread
import re
from bs4 import BeautifulSoup
import requests
import json # To handle service account JSON

# --- Configuration ---
# You'll replace this with your actual Google Sheet URL
GOOGLE_SHEET_URL = "YOUR_GOOGLE_SHEET_URL_HERE" # e.g., "https://docs.google.com/spreadsheets/d/1Bgi710Gg4Vb9F_g9oW5jA_g9oW5jA_g9oW5jA_g9oW5jA/edit#gid=0"
SERVICE_ACCOUNT_FILE = "service_account.json" # Make sure this file is in your project directory

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to reduce API calls
def load_data_from_google_sheets():
    """Loads product data from the Google Sheet."""
    try:
        # Authenticate with Google Sheets using the service account
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        
        # Open the Google Sheet by its URL or title
        # If using URL, ensure it's the full URL
        # If using title, ensure your sheet title is unique
        sh = gc.open_by_url(GOOGLE_SHEET_URL) # or gc.open("Butler Coffee Lab Products Sheet")
        
        # Select the first worksheet
        worksheet = sh.worksheet("Sheet1") # Replace "Sheet1" with your actual sheet name if different
        
        # Get all records as a list of dictionaries
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Convert column names to a consistent format (e.g., lowercase, no spaces)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        # Ensure essential columns exist
        required_cols = ['name', 'category_/_type', 'short_description', 'long_description', 'price', 'bcl_website_link']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column in Google Sheet: '{col}'. Please check your sheet headers.")
                return pd.DataFrame() # Return empty DataFrame on error
        
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.info("Please ensure your `service_account.json` is correct and the sheet is shared with the service account email.")
        return pd.DataFrame()

def generate_tags(description):
    """Generates a list of tags from product descriptions."""
    # This is a basic example. You can expand this with more sophisticated NLP
    # or a predefined list of keywords specific to coffee/tea flavors.
    keywords = [
        "chocolate", "nutty", "smooth", "fruity", "caramel", "bold", "bright",
        "citrus", "floral", "spicy", "earthy", "roasty", "sweet", "mellow",
        "velvet", "crisp", "clean", "rich", "balanced", "smoky", "cacao",
        "berry", "apple", "peach", "vanilla", "honey", "maple", "toffee",
        "grapefruit", "lemon", "lime", "almond", "hazelnut", "pecan", "walnut",
        "cherry", "plum", "currant", "dark chocolate", "milk chocolate", "cocoa"
    ]
    
    found_tags = []
    text = description.lower()
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            found_tags.append(keyword)
    return list(set(found_tags)) # Return unique tags

@st.cache_data(ttl=3600*24) # Cache images for a day
def scrape_image(url):
    """Scrapes the main product image from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Common selectors for product images. You might need to inspect BCL's site specifically.
        # Look for <img class="product-image"> or <img data-image-id="main-product-image"> etc.
        img_tag = soup.find('img', class_=lambda x: x and ('product-image' in x or 'main-image' in x))
        if not img_tag:
             img_tag = soup.find('meta', property='og:image') # Try Open Graph meta tag
             if img_tag:
                 return img_tag.get('content')
        
        if img_tag and img_tag.get('src'):
            # Ensure the URL is absolute
            src = img_tag.get('src')
            if not src.startswith(('http', 'https')):
                # Attempt to construct absolute URL if relative
                base_url = url.split('/')[0] + '//' + url.split('/')[2]
                src = f"{base_url}{src}" if src.startswith('/') else f"{base_url}/{src}"
            return src
        elif img_tag and img_tag.get('data-src'): # Sometimes images use data-src
             src = img_tag.get('data-src')
             if not src.startswith(('http', 'https')):
                base_url = url.split('/')[0] + '//' + url.split('/')[2]
                src = f"{base_url}{src}" if src.startswith('/') else f"{base_url}/{src}"
             return src
        
        st.warning(f"Could not find a suitable image for {url}. Please check the URL or provide a more specific selector.")
        return None # No image found
    except requests.exceptions.RequestException as e:
        st.error(f"Error scraping image from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping image from {url}: {e}")
        return None


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
    # --- Generate Tags (if not already present or if you want to re-generate) ---
    # This assumes 'long_description' is the best source for tags.
    # Adjust column name if necessary.
    df_products['tags'] = df_products['long_description'].apply(generate_tags)
    
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

        # Only show roast preference if Coffee is selected
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
        # --- AI-Powered Recommendations Logic (Placeholder) ---
        # This is where your matching logic goes.
        # For now, it's a simple filter based on basic criteria.

        filtered_products = df_products[
            (df_products['category_/_type'].str.contains(drink_type, case=False, na=False)) &
            (df_products['brew_method'].fillna('').apply(lambda x: any(b in x for b in brew_method))) # Assuming 'brew_method' column exists and is a string of methods
        ]

        if drink_type == "Coffee" and roast_preference != "No preference":
            filtered_products = filtered_products[
                filtered_products['roast_level'].str.contains(roast_preference, case=False, na=False) # Assuming 'roast_level' column exists
            ]
        
        if surprise_me:
            if not filtered_products.empty:
                recommendations = filtered_products.sample(min(5, len(filtered_products)))
            else:
                recommendations = pd.DataFrame()
        else:
            # Basic flavor matching logic
            if flavor_input:
                user_flavor_tags = [tag.strip().lower() for tag in flavor_input.split(',')]
                
                # Filter by matching any of the user's input tags
                # This logic can be greatly improved using similarity measures (e.g., cosine similarity)
                # with more advanced NLP embeddings if you want "AI" flavor matching.
                def matches_flavor(product_tags, user_tags):
                    if not product_tags: return False
                    return any(ut in pt for ut in user_tags for pt in product_tags)

                recommendations = filtered_products[
                    filtered_products['tags'].apply(lambda x: matches_flavor(x, user_flavor_tags))
                ]
                
                if recommendations.empty:
                    st.warning("No direct matches found for your flavor preferences. Showing some general recommendations.")
                    # Fallback to general filtered products if no flavor match
                    recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()
            else:
                # If no flavor input and not surprise me, show some general picks
                recommendations = filtered_products.sample(min(3, len(filtered_products))) if not filtered_products.empty else pd.DataFrame()

        # --- Display Results ---
        if not recommendations.empty:
            st.markdown("---")
            st.markdown('<p class="big-font">Your Personalized Butler Coffee Lab Picks:</p>', unsafe_allow_html=True)

            # Flavor Profile Summary
            st.markdown("""
            <p style="font-size:16px; color:#555555;">
            You love mellow mornings with notes of chocolate and caramel.
            This lineup is cozy, smooth, and just a little nutty—like your perfect Sunday.
            Each sip supports inclusive employment. <a href="#" style="color:#B35F3A; font-weight:bold;">Donate here!</a>
            </p>
            """, unsafe_allow_html=True)


            for _, product in recommendations.iterrows():
                product_name = product.get('name', 'N/A')
                short_desc = product.get('short_description', product.get('long_description', 'A delightful product.')[:100] + '...')
                price = product.get('price', 'N/A')
                bcl_link = product.get('bcl_website_link', '#')
                product_tags = ", ".join(product.get('tags', []))

                # Scrape image (this might be slow, consider pre-scraping or using a CDN)
                image_url = scrape_image(bcl_link) if bcl_link and bcl_link != '#' else None
                
                st.markdown(f"""
                <div class="product-card">
                    {"<img src='" + image_url + "' alt='" + product_name + "' />" if image_url else ""}
                    <div class="product-card-details">
                        <h4>{product_name}</h4>
                        <p>{short_desc}</p>
                        <p class="price">${price}</p>
                        <p class="tags">Flavor Tags: {product_tags if product_tags else "None"}</p>
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
