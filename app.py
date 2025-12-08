import streamlit as st
import google.generativeai as genai
import PyPDF2
import pandas as pd
import io
import time
import json
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="G.O.A.A. Sales Auditor (Live)", page_icon="⚡", layout="wide")

# --- SESSION STATE SETUP ---
if "results_list" not in st.session_state:
    st.session_state.results_list = []
if "processing_active" not in st.session_state:
    st.session_state.processing_active = False

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

    **YOUR TASK: Extract the following fields based on the user's specific requirements.**

    **1. Customer Priming (Yes/No):**
       Check if the customer was aware of these BEFORE the core pitch started (did they mention being told by a pre-sales person?):
       - Price Rise Awareness
       - Limited Inventory Awareness
       - Value of attending this call
       - Time-sensitive window

    **2. Motivation & Tailoring:**
       - **Motivation Checked?** (Yes/No)
       - **Identified Motivation:** (e.g., Rental Yield, Holiday Home, Capital Appreciation).
       - **Tailored Pitch?** Did the CSM adapt the pitch based on this motivation? (Yes/No).

    **3. Objections (Bucketing):**
       Mark "Yes" if the customer raised an objection in these categories:
       - Price Related
       - Product Related (Size, specs)
       - Location Related (Bicholim, distance from beach)
       - ROI Related (Yield, growth)
       - Site Visit Related (Want to see before buying)
       - Payment Terms Related

    **4. Q&A Log (Verbatim):**
       List the specific questions/objections and the CSM's answers.
       *Format:* "Cust: [Objection translated to English] -> CSM: [Answer translated to English]"

    **5. Urgency Creation:**
       - **Urgency Established?** (Yes/No)
       - **Closing Remarks/Tactics:** Describe how they pushed for the EOI/Application (e.g., "Only 2 units left", "Price increases tomorrow").

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
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    csv_data = df.to_csv(index=False)
    prompt = f"""
    You are the Sales Training Head at HoABL. Summarize these G.O.A.A. sales calls.
    Data: {csv_data}
    **STRICT RULE:** 100% English Output.
    **TASK 1: CSM SUMMARY** (JSON list of objects: "CSM Name", "Strengths", "Areas of Improvement", "Specific Instances")
    **TASK 2: TEAM SUMMARY** (JSON list of strings: "Team Performance Insights")
    Return strictly Valid JSON: {{ "CSM_Summaries": [], "Team_Summary": [] }}
    """
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(response.text.strip())
        return pd.DataFrame(data.get("CSM_Summaries", [])), pd.DataFrame(data.get("Team_Summary", []), columns=["Team Performance Insights"])
    except:
        return pd.DataFrame(), pd.DataFrame()

# --- MAIN UI ---
st.title("⚡ G.O.A.A. Live Sales Auditor")

with st.sidebar:
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_files = st.file_uploader("Upload Transcripts (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Start Processing"):
        st.session_state.results_list = [] # Reset on new run
        st.session_state.processing_active = True

# --- LIVE DASHBOARD ---
status_container = st.container()
table_container = st.empty()
summary_container = st.container()

if st.session_state.processing_active and uploaded_files and api_key:
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        progress_bar = status_container.progress(0)
        status_text = status_container.empty()
        
        # MAIN LOOP
        for i, file in enumerate(uploaded_files):
            # Check if already processed (duplicates)
            if any(d['File Name'] == file.name for d in st.session_state.results_list):
                continue
                
            status_text.text(f"⏳ Analyzing: {file.name} ({i+1}/{len(uploaded_files)})")
            
            text = extract_text_from_pdf(file)
            if text:
                res = analyze_single_call(model, text)
                if res:
                    res['File Name'] = file.name
                    st.session_state.results_list.append(res)
                    
                    # LIVE UPDATE
                    df_live = pd.DataFrame(st.session_state.results_list)
                    table_container.dataframe(df_live, use_container_width=True)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            time.sleep(0.1) # Brief pause for UI refresh

        status_text.success("✅ All files processed!")
        st.session_state.processing_active = False # Stop processing state
        
        # GENERATE SUMMARIES AFTER LOOP
        with st.spinner("🧠 Generating Management Reports..."):
            df_final = pd.DataFrame(st.session_state.results_list)
            df_csm, df_team = generate_summaries(model, df_final)
            
            # Display Summaries
            with summary_container:
                tab_a, tab_b = st.tabs(["CSM Summary", "Team Stats"])
                tab_a.dataframe(df_csm, use_container_width=True)
                tab_b.dataframe(df_team, use_container_width=True)

            # DOWNLOAD BUTTON
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final.to_excel(writer, sheet_name='Analysis', index=False)
                df_csm.to_excel(writer, sheet_name='CSM Summary', index=False)
                df_team.to_excel(writer, sheet_name='Team Stats', index=False)
            
            st.download_button(
                label="📥 Download Complete Report",
                data=output.getvalue(),
                file_name="GOAA_Full_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        status_container.error(f"Error: {e}")

# --- FALLBACK DISPLAY (If page refreshes but data exists) ---
elif st.session_state.results_list:
    st.info("Showing previous results.")
    st.dataframe(pd.DataFrame(st.session_state.results_list))
