import streamlit as st
import pandas as pd
import io
import time
import json
import re
import os
import subprocess
import sys

# --- AUTO-FIX FOR OLD LIBRARIES ---
# This ensures the app doesn't crash on Streamlit Cloud or Local
try:
    import google.generativeai as genai
    # Check if 'configure' exists. If not, trigger update.
    if not hasattr(genai, 'configure'):
        raise ImportError("Old version detected")
except ImportError:
    st.warning("⚠️ Updating Google AI Library... (This happens once)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "google-generativeai"])
    import google.generativeai as genai
    st.success("✅ Library updated! Please rerun the app.")
    st.stop()

import PyPDF2

# --- CONFIGURATION ---
st.set_page_config(page_title="G.O.A.A. Sales Auditor", page_icon="🏖️", layout="wide")

# --- SESSION STATE SETUP ---
if "results_list" not in st.session_state:
    st.session_state.results_list = []
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False
if "df_csm" not in st.session_state:
    st.session_state.df_csm = None
if "df_team" not in st.session_state:
    st.session_state.df_team = None

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except: return ""

def analyze_single_call(model, text):
    prompt = """
    You are a QA Auditor for 'The House of Abhinandan Lodha' (HoABL) specifically for the project 'G.O.A.A. Premium Residences'.
    Analyze this sales call transcript.

    **STRICT LANGUAGE RULE:** 1. THE OUTPUT MUST BE 100% ENGLISH. 
    2. DO NOT USE HINDI SCRIPT (Devanagari). 
    3. If the transcript contains Hindi, TRANSLATE the specific quotes into English.

    **REFERENCE DOCUMENT HIGHLIGHTS (THE "GUIDE"):**
    1. **Project:** G.O.A.A. Premium Residences, Bicholim (North Goa). 40 mins from MOPA Airport.
    2. **Product:** 1 BHK & 2 BHK Premium Serviced Residences. Serviced by MIROS Hotels & Resorts (5-Star).
    3. **USP:** Man-made sea & beach, 130+ acre resort ecosystem, high rental yield (~8% for 1BHK).
    4. **Offers:** Club Membership Waiver (~10L), Spot Offer (70k), Corpus Waiver (1.30L), Payment Plan (25:25:25:25).
    5. **Growth:** 3X appreciation in 7 years (Colliers Report).

    **YOUR TASK: Extract the following fields.**

    **1. Customer Priming (Yes/No):**
       Was the customer told about these BEFORE the core pitch started?
       - Price Rise Awareness
       - Limited Inventory Awareness
       - Value of attending this call
       - Time-sensitive window

    **2. Motivation & Tailoring:**
       - **Motivation Checked?** (Yes/No)
       - **Identified Motivation:** (e.g., Rental Yield, Holiday Home, Appreciation).
       - **Tailored Pitch?** Did the CSM adapt the pitch based on this motivation? (Yes/No).

    **3. Objections (Bucketing):**
       Mark "Yes" if raised:
       - Price Related
       - Product Related (Size, specs)
       - Location Related (Bicholim, distance)
       - ROI Related (Yield, growth)
       - Site Visit Related
       - Payment Terms Related

    **4. Q&A Log (Verbatim):**
       List specific questions/objections and answers.
       *Format:* "Cust: [English Translation] -> CSM: [English Translation]"

    **5. Urgency Creation:**
       - **Urgency Established?** (Yes/No)
       - **Closing Remarks:** How did they push for the EOI?

    **OUTPUT FORMAT:**
    Return a SINGLE line separated by '###' delimiters containing exactly these 18 fields:
    CSM Name###Customer Name###Primed: Price Rise (Y/N)###Primed: Inventory (Y/N)###Primed: Call Value (Y/N)###Primed: Time Sensitive (Y/N)###Motivation Checked (Y/N)###Customer Motivation###Pitch Tailored (Y/N)###Obj: Price (Y/N)###Obj: Product (Y/N)###Obj: Location (Y/N)###Obj: ROI (Y/N)###Obj: Site Visit (Y/N)###Obj: Payment (Y/N)###Verbatim Q&A###Urgency Created (Y/N)###Closing Remarks/Urgency Tactic

    **TRANSCRIPT:**
    """ + text[:30000]

    try:
        response = model.generate_content(prompt)
        parts = [x.strip() for x in response.text.strip().split('###')]
        while len(parts) < 18: parts.append("-")
        return {
            "CSM Name": parts[0], "Customer Name": parts[1],
            "Primed: Price Rise": parts[2], "Primed: Inventory": parts[3],
            "Primed: Call Value": parts[4], "Primed: Time Sens.": parts[5],
            "Motivation Checked?": parts[6], "Customer Motivation": parts[7],
            "Pitch Tailored?": parts[8],
            "⚠️ Obj: Price": parts[9], "⚠️ Obj: Product": parts[10],
            "⚠️ Obj: Location": parts[11], "⚠️ Obj: ROI": parts[12],
            "⚠️ Obj: Site Visit": parts[13], "⚠️ Obj: Payment": parts[14],
            "📝 Verbatim Q&A": parts[15], "Urgency Created?": parts[16],
            "Closing/Urgency Tactic": parts[17]
        }
    except: return None

def generate_summaries(model, df):
    if df.empty: return None, None
    csv_data = df.to_csv(index=False)
    
    prompt = f"""
    You are the Sales Training Head at HoABL. Summarize these G.O.A.A. sales calls.
    Data: {csv_data}
    
    **STRICT RULE:** 100% English Output. No Hindi.

    **TASK 1: CSM SUMMARY** (JSON list of objects: "CSM Name", "Strengths", "Areas of Improvement", "Specific Instances")
    **TASK 2: TEAM SUMMARY** (JSON list of strings: "Team Performance Insights")

    Return strictly Valid JSON: {{ "CSM_Summaries": [], "Team_Summary": [] }}
    """
    try:
        response = model.generate_content(prompt)
        # Robust JSON Extraction
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        json_str = match.group(0) if match else text.replace("```json", "").replace("```", "")
        data = json.loads(json_str)
        
        return pd.DataFrame(data.get("CSM_Summaries", [])), pd.DataFrame(data.get("Team_Summary", []), columns=["Team Performance Insights"])
    except:
        return None, None

# --- MAIN UI ---
st.title("🏖️ G.O.A.A. Sales Auditor")

with st.sidebar:
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_files = st.file_uploader("Upload Transcripts (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Start Processing"):
        st.session_state.results_list = [] 
        st.session_state.df_csm = None
        st.session_state.df_team = None
        st.session_state.processing_active = True

# --- PROCESS LOGIC ---
status_container = st.container()
tab1, tab2, tab3 = st.tabs(["Analysis (Live)", "CSM Summary", "Team Stats"])
table_placeholder = tab1.empty()

if st.session_state.processing_active and uploaded_files and api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        progress_bar = status_container.progress(0)
        status_text = status_container.empty()
        
        # 1. ANALYZE FILES
        for i, file in enumerate(uploaded_files):
            # Skip duplicates
            if any(d.get('File Name') == file.name for d in st.session_state.results_list):
                continue
                
            status_text.text(f"⏳ Analyzing: {file.name}...")
            text = extract_text_from_pdf(file)
            
            if text:
                res = analyze_single_call(model, text)
                if res:
                    res['File Name'] = file.name
                    st.session_state.results_list.append(res)
                    
                    # Update Table Live
                    table_placeholder.dataframe(pd.DataFrame(st.session_state.results_list))
            
            # Update Progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            time.sleep(0.1) # Fast processing for Tier 1

        st.session_state.processing_active = False 
        status_text.success("✅ File Analysis Complete! Generating Summaries...")
        
        # 2. GENERATE SUMMARIES
        df_final = pd.DataFrame(st.session_state.results_list)
        df_csm, df_team = generate_summaries(model, df_final)
        
        st.session_state.df_csm = df_csm
        st.session_state.df_team = df_team
        status_text.empty() # Clear status
        progress_bar.empty()

    except Exception as e:
        status_container.error(f"Error: {e}")

# --- DISPLAY RESULTS ---
if st.session_state.results_list:
    # Tab 1 is updated live above, forcing refresh here just in case
    table_placeholder.dataframe(pd.DataFrame(st.session_state.results_list))

    with tab2:
        if st.session_state.df_csm is not None:
            st.dataframe(st.session_state.df_csm, use_container_width=True)
        else:
            st.info("Waiting for completion...")

    with tab3:
        if st.session_state.df_team is not None:
            st.dataframe(st.session_state.df_team, use_container_width=True)
        else:
            st.info("Waiting for completion...")

    # EXCEL DOWNLOAD
    if st.session_state.df_csm is not None:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(st.session_state.results_list).to_excel(writer, sheet_name='Analysis', index=False)
            st.session_state.df_csm.to_excel(writer, sheet_name='CSM Summary', index=False)
            st.session_state.df_team.to_excel(writer, sheet_name='Team Stats', index=False)
        
        st.download_button(
            label="📥 Download Final Report",
            data=output.getvalue(),
            file_name="GOAA_Sales_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
