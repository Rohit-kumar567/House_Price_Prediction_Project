import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Currency conversion rates (as of January 2026 - approximate)
# Base: 1 USD equals...
CURRENCY_RATES = {
    # Americas
    "USD ($) - United States": 1.0,
    "CAD (C$) - Canada": 1.35,
    "MXN ($) - Mexico": 17.20,
    "BRL (R$) - Brazil": 4.95,
    "ARS ($) - Argentina": 850.00,

    # Europe
    "EUR (€) - Eurozone": 0.92,
    "GBP (£) - United Kingdom": 0.79,
    "CHF (Fr) - Switzerland": 0.88,
    "SEK (kr) - Sweden": 10.35,
    "NOK (kr) - Norway": 10.60,
    "DKK (kr) - Denmark": 6.85,
    "PLN (zł) - Poland": 4.05,
    "CZK (Kč) - Czech Republic": 22.50,
    "HUF (Ft) - Hungary": 355.00,
    "RUB (₽) - Russia": 92.00,
    "TRY (₺) - Turkey": 32.50,

    # Asia
    "INR (₹) - India": 83.50,
    "CNY (¥) - China": 7.24,
    "JPY (¥) - Japan": 149.20,
    "KRW (₩) - South Korea": 1320.00,
    "SGD (S$) - Singapore": 1.34,
    "HKD (HK$) - Hong Kong": 7.81,
    "THB (฿) - Thailand": 34.50,
    "MYR (RM) - Malaysia": 4.45,
    "IDR (Rp) - Indonesia": 15750.00,
    "PHP (₱) - Philippines": 56.50,
    "VND (₫) - Vietnam": 24500.00,
    "PKR (₨) - Pakistan": 278.00,
    "BDT (৳) - Bangladesh": 110.00,
    "LKR (Rs) - Sri Lanka": 305.00,
    "NPR (₨) - Nepal": 133.50,

    # Middle East & Africa
    "AED (د.إ) - UAE": 3.67,
    "SAR (﷼) - Saudi Arabia": 3.75,
    "ILS (₪) - Israel": 3.65,
    "ZAR (R) - South Africa": 18.50,
    "EGP (£) - Egypt": 48.50,
    "NGN (₦) - Nigeria": 1450.00,
    "KES (KSh) - Kenya": 127.50,

    # Oceania
    "AUD (A$) - Australia": 1.52,
    "NZD (NZ$) - New Zealand": 1.65,
}

# Currency symbols for clean display
CURRENCY_SYMBOLS = {
    # Americas
    "USD ($) - United States": "$",
    "CAD (C$) - Canada": "C$",
    "MXN ($) - Mexico": "MX$",
    "BRL (R$) - Brazil": "R$",
    "ARS ($) - Argentina": "AR$",

    # Europe
    "EUR (€) - Eurozone": "€",
    "GBP (£) - United Kingdom": "£",
    "CHF (Fr) - Switzerland": "Fr",
    "SEK (kr) - Sweden": "kr",
    "NOK (kr) - Norway": "kr",
    "DKK (kr) - Denmark": "kr",
    "PLN (zł) - Poland": "zł",
    "CZK (Kč) - Czech Republic": "Kč",
    "HUF (Ft) - Hungary": "Ft",
    "RUB (₽) - Russia": "₽",
    "TRY (₺) - Turkey": "₺",

    # Asia
    "INR (₹) - India": "₹",
    "CNY (¥) - China": "¥",
    "JPY (¥) - Japan": "¥",
    "KRW (₩) - South Korea": "₩",
    "SGD (S$) - Singapore": "S$",
    "HKD (HK$) - Hong Kong": "HK$",
    "THB (฿) - Thailand": "฿",
    "MYR (RM) - Malaysia": "RM",
    "IDR (Rp) - Indonesia": "Rp",
    "PHP (₱) - Philippines": "₱",
    "VND (₫) - Vietnam": "₫",
    "PKR (₨) - Pakistan": "₨",
    "BDT (৳) - Bangladesh": "৳",
    "LKR (Rs) - Sri Lanka": "Rs",
    "NPR (₨) - Nepal": "₨",

    # Middle East & Africa
    "AED (د.إ) - UAE": "AED",
    "SAR (﷼) - Saudi Arabia": "SAR",
    "ILS (₪) - Israel": "₪",
    "ZAR (R) - South Africa": "R",
    "EGP (£) - Egypt": "E£",
    "NGN (₦) - Nigeria": "₦",
    "KES (KSh) - Kenya": "KSh",

    # Oceania
    "AUD (A$) - Australia": "A$",
    "NZD (NZ$) - New Zealand": "NZ$",
}

# Area unit conversion rates (1 sq ft equals...)
AREA_UNITS = {
    "Square Feet (sq ft)": 1.0,
    "Square Meters (m²)": 0.092903,
    "Square Yards (sq yd)": 0.111111,
    "Acres": 0.000022956,
    "Cents (South India)": 0.002296,
    "Guntha (Maharashtra)": 0.00036734,
    "Bigha (1 acre)": 0.000022956,
    "Hectares": 0.0000092903,
}

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# To show the title and display an text
st.title("🏠 House Price Prediction App")
st.write("Enter the house size to predict its price!")


# Load the model
@st.cache_resource  # It will load the model once and remember it
def load_model():
    model = joblib.load("house_price_model.pkl")  # Loads your saved trained model
    return model  # Returns the model


model = load_model()

# Now splitting the page into two parts
col1, col2 = st.columns(2)

# Use the left part of the area as the input part
with col1:
    st.subheader("📊 Input features")  # Sub-heading

    # house_size is the variable that holds the value
    house_size = st.slider(
        "House Size(sq ft)",  # text message
        min_value=500,  # Slider min value
        max_value=3500,  # Sliders max value
        value=2000,  # Slider starts from
        step=50  # Steps it moves
    )

    # Unit selector
    selected_unit = st.selectbox(
        "📏 Display Size In",
        options=list(AREA_UNITS.keys()),
        index=0  # Default to sq ft
    )

    # Convert house size to selected unit
    conversion_factor = AREA_UNITS[selected_unit]
    house_size_converted = house_size * conversion_factor

    # Dispaly the size which user selected
    st.metric("Selected Size", f"{house_size_converted:,.4f} {selected_unit}")

    st.markdown("---")

    # Currency selector
    selected_currency = st.selectbox(
        "💱 Select Currency",
        options=list(CURRENCY_RATES.keys()),
        index=16  # This is INR's position (currently index=1 is CAD)
    )

    # Creating a clickable button
    if st.button("🔮 Predict Price", type="primary"):  # it runs when the button is clicked
        prediction = model.predict([[house_size]])  # Use the model to predict the price
        predicted_price_usd = prediction[0]  # Gets the actual price value

        # Get conversion rate and symbol for selected currency
        conversion_rate = CURRENCY_RATES[selected_currency]
        currency_symbol = CURRENCY_SYMBOLS[selected_currency]

        # Convert to selected currency
        predicted_price_converted = predicted_price_usd * conversion_rate

        # Professional dashboard style output
        st.markdown("---")

        # Header
        st.markdown("### 🎯 Price Analysis")

        # Main price display
        st.markdown(f"""
        <div style='background-color: #f8f9fa; 
                    padding: 20px; 
                    border-left: 5px solid #4CAF50; 
                    border-radius: 5px; 
                    margin: 10px 0;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>ESTIMATED HOUSE PRICE</p>
            <h2 style='color: #2c3e50; margin: 5px 0;'>{currency_symbol}{predicted_price_converted:,.2f}</h2>
            <p style='color: #888; margin: 0; font-size: 12px;'>{selected_currency}</p>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: House Size (full width)
        st.markdown(f"""
        <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;'>
            <p style='color: #1976d2; margin: 0; font-size: 14px; font-weight: bold;'>HOUSE SIZE</p>
            <h2 style='color: #0d47a1; margin: 10px 0;'>{house_size_converted:,.4f}</h2>
            <p style='color: #1976d2; margin: 0; font-size: 16px;'>{selected_unit}</p>
        </div>
        """, unsafe_allow_html=True)

        # Row 2: Price per unit and USD Value (side by side)
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            price_per_unit = predicted_price_converted / house_size_converted if house_size_converted > 0 else 0
            unit_short = selected_unit.split('(')[0].strip()

            if price_per_unit >= 10000000:
                value_display = f"{currency_symbol}{price_per_unit / 10000000:.2f}Cr"
            elif price_per_unit >= 100000:
                value_display = f"{currency_symbol}{price_per_unit / 100000:.2f}L"
            elif price_per_unit >= 1000:
                value_display = f"{currency_symbol}{price_per_unit / 1000:.2f}K"
            else:
                value_display = f"{currency_symbol}{price_per_unit:,.0f}"

            st.markdown(f"""
            <div style='background-color: #fff3e0; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #e65100; margin: 0; font-size: 12px;'>PRICE/{unit_short.upper()}</p>
                <h3 style='color: #bf360c; margin: 10px 0;'>{value_display}</h3>
            </div>
            """, unsafe_allow_html=True)

        with detail_col2:
            label = "USD VALUE" if selected_currency != "USD ($) - United States" else "TOTAL VALUE"
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center;'>
                <p style='color: #2e7d32; margin: 0; font-size: 12px;'>{label}</p>
                <h3 style='color: #1b5e20; margin: 10px 0;'>${predicted_price_usd:,.0f}</h3>
            </div>
            """, unsafe_allow_html=True)

    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset", type="secondary"):
        st.rerun()

# Now Right side column
with col2:
    st.subheader("📈 Model Information")

    st.write("**Model Information:**")
    st.write(f"- Coefficient (Slope): {model.coef_[0]:.2f}")
    st.write(f"- Intercept: {model.intercept_:.2f}")

    st.latex(f"Price = {model.coef_[0]:.2f} \\times Size + {model.intercept_:.2f}")

st.markdown("---")
st.markdown("**Built with Streamlit** | Machine Learning House Price Predictor")

