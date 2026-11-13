import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒")

@st.cache_resource
def load():
    model = joblib.load("model.joblib")
    cfg = json.load(open("app_config.json"))
    return model, cfg

model, cfg = load()
defaults, columns = cfg["defaults"], cfg["columns"]

st.title("🛒 Will This Visitor Buy?")
st.write("Enter a browsing session and get the purchase probability and a suggested action.")

with st.sidebar:
    st.header("Action thresholds")
    st.caption("Business rule: you can change these.")
    t_high = st.slider("No action if probability ≥", 0.1, 0.9, 0.5, 0.05)
    t_low = st.slider("Show chat if probability <", 0.01, 0.5, 0.15, 0.01)

c1, c2 = st.columns(2)
with c1:
    page_values = st.number_input("PageValues", 0.0, 500.0, 0.0, help="Average value of pages visited before purchase")
    exit_rates = st.number_input("ExitRates", 0.0, 1.0, 0.04, 0.01)
    bounce_rates = st.number_input("BounceRates", 0.0, 1.0, 0.02, 0.01)
    special_day = st.selectbox("SpecialDay (closeness to a special day)", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    weekend = st.checkbox("Weekend visit")
with c2:
    product_pages = st.number_input("Product pages visited", 0, 1000, 20)
    product_dur = st.number_input("Seconds on product pages", 0.0, 100000.0, 600.0)
    admin_pages = st.number_input("Administrative pages visited", 0, 100, 2)
    admin_dur = st.number_input("Seconds on administrative pages", 0.0, 10000.0, 50.0)
    info_pages = st.number_input("Informational pages visited", 0, 100, 0)
    info_dur = st.number_input("Seconds on informational pages", 0.0, 10000.0, 0.0)

month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
visitor = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])

if st.button("Predict", type="primary"):
    row = dict(defaults)
    row.update({
        "PageValues": page_values, "ExitRates": exit_rates, "BounceRates": bounce_rates,
        "SpecialDay": special_day, "Weekend": int(weekend),
        "ProductRelated": product_pages, "ProductRelated_Duration": product_dur,
        "Administrative": admin_pages, "Administrative_Duration": admin_dur,
        "Informational": info_pages, "Informational_Duration": info_dur,
        "Month": month, "VisitorType": visitor,
    })
    X = pd.DataFrame([row])[columns]
    p = float(model.predict_proba(X)[0, 1])

    st.metric("Purchase probability", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= t_high:
        st.success("Likely to buy → **No action** (don't give a discount).")
    elif p >= t_low:
        st.warning("Undecided → **Show a coupon**.")
    else:
        st.info("Unlikely to buy → **Offer the chat assistant**.")
    st.caption("This is a model estimate, not a guarantee. It was trained on one store's data.")
