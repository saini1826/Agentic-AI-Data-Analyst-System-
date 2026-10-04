import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    from groq import Groq
except ImportError:
    Groq = None

st.set_page_config(page_title='analystOS')
st.title('AnalystOS')

api_key = st.sidebar.text_input('groq api key', type='password')
model = st.sidebar.selectbox('model',["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"])

if api_key and Groq:
    st.sidebar.success('connected')

def ask(prompt):
    if not api_key or not Groq:
        return "kindly enter the api key"
    try:
        r = Groq(api_key=api_key).chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'you are a concise data analyst'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.35,
            max_tokens=900
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Error:{e}"

# upload
file = st.file_uploader('only takes csv file', type='csv')
if not file:
    st.stop()

df = pd.read_csv(file, encoding="latin1", encoding_errors="ignore")
cols, missing = list(df.columns), df.isnull().sum()
summary = df.describe(include='all').to_string()[:800]
num_cols = df.select_dtypes(include='number').columns.tolist()
sample = df.head(6).to_string(index=False)

st.success(f"{file.name} - {df.shape[0]:,} rows, {df.shape[1]} columns")

# metrix
c1, c2, c3, c4 = st.columns(4)
for c, lbl, val in [(c1, "rows", f"{df.shape[0]:,}"), (c2, "columns", df.shape[1]), (c3, "missing", int(missing.sum())),(c4,"numeric cols",len(num_cols))]:
    c.metric(lbl, val)

with st.expander('preview & column info'):
    st.dataframe(df.head(8), use_container_width=True)
    st.dataframe(pd.DataFrame({"type": df.dtypes.astype(str), "missing": missing, "Unique": df.nunique()}),use_container_width=True)

tabs = st.tabs(["Domain", "Cleaning", "Insights", "Business", "CEO Report"])
prompts = [
    f"Dataset domain kya hai (Sales/HR/Finance/Healthcare/Ecommerce/Other)?\nColumns: {cols}",
    f"Short bullet-point cleaning tips do.\nColumns: {cols}\nMissing:\n{missing}",
    f"5 key insights bullet points mein do. Column names aur numbers use karo.\nSummary:\n{summary}",
    f"Business analysis do - Risks, Opportunities, Recommendations (2-3 each).\nColumns: {cols}",
    f"CEO report likho - Executive Summary, Key Trends, Risks, Opportunities, Recommendations.\nSummary:\n{summary}"
]
btn_labels = ["Detect Domain", "Get Cleaning Tips", "Find Insights", "Run Analysis", "CEO Report"]

for i, (tab, prompt, btn) in enumerate(zip(tabs, prompts, btn_labels)):
    with tab:
        key = f"res_{i}"
        if key not in st.session_state:
            st.session_state[key] = ""
        if st.button(btn, key=f'btn_{i}'):
            with st.spinner("processing.."):
                st.session_state[key] = ask(prompt)
        if st.session_state[key]:
            st.write(st.session_state[key])
            if i == 4:
                st.download_button("download report", st.session_state[key], "CEO_report.txt")

# chart
st.subheader('charts')
if not num_cols:
    st.info('numeric nhi hai ')
else:
    ct1, ct2 = st.tabs(["histogram", "scatter"])
    with ct1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df[num_cols[0]].dropna(), bins=20, color='#3b6fd4', edgecolor='#2a56b0', linewidth=0.5)
        st.pyplot(fig, use_container_width=True)
    with ct2:
        if len(num_cols) < 2:
            st.info("kam se kam 2 numeric col chahiye")
        else:
            cx, cy = st.columns(2)
            x = cx.selectbox("x", num_cols, 0, key="sx")
            y = cy.selectbox("y", num_cols, 1, key="sy")
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            ax2.scatter(df[x].dropna()[:2000], df[y].dropna()[:2000], alpha=0.5, s=18,color="#e05c2a")
            st.pyplot(fig2, use_container_width=True)

st.subheader('koi bhi sawala pucho')
q = st.text_input("sawal...", label_visibility='collapsed', placeholder="e.g:- which category performs best")
if st.button('pucho', key='btn_qa'):
    if q.strip():
        with st.spinner('soch raha hoon'):
            st.write(ask(f"columns:{cols}\nsummary:\n{summary}\nquestion:{q}\nanswer concisely,"))
    else:
        st.warning('pehle sawal likho')