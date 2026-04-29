import streamlit as st
import pandas as pd
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
df = pd.read_csv("ghana_extracted.csv")  # put CSV in same folder as app.py

def df_to_context(df):
    records = []
    for _, row in df.iterrows():
        record = f"""
Facility: {row.get('name', '')}
City: {row.get('address_city', '')}
Region: {row.get('address_stateOrRegion', '')}
Type: {row.get('facilityTypeId', '')}
Doctors: {row.get('numberDoctors', '')}
Capacity: {row.get('capacity', '')}
Specialties: {row.get('specialties', '')}
Procedures: {row.get('procedure', '')}
Equipment: {row.get('equipment', '')}
Capabilities: {row.get('capability', '')}
"""
        records.append(record)
    return "\n---\n".join(records)

DATASET_CONTEXT = df_to_context(df)

SYSTEM_PROMPT = f"""
You are a medical facility intelligence assistant for Ghana.
Answer questions about these facilities accurately and concisely.
If asked to list facilities, always include their name and city.
If no facilities match, say so clearly.

FACILITY DATA:
{DATASET_CONTEXT}
"""

st.set_page_config(page_title="Ghana Medical Facility Assistant", page_icon="🏥")
st.title("🏥 Ghana Medical Facility Assistant")
st.caption("Ask questions about healthcare facilities in Ghana in plain English.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g. Which hospitals in Accra have an ICU?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *st.session_state.messages
                ],
                temperature=0
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
