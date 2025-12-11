import streamlit as st
import google.generativeai as genai
import PyPDF2
import pandas as pd
import io
import time
import json
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="G.O.A.A. Sales Auditor (Debug Mode)", page_icon="🛠️", layout="wide")

# --- SESSION STATE ---
if "results_list" not in st.session_state:
    st.session_state.results_list = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- HELPER FUNCTIONS ---
def log_message(msg):
    st.session_state.logs.append(msg)
    # Force UI update for log (optional, but keeps it reactive)

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return None

def analyze_single_call(model, text, filename):
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
       - Price Rise Awareness
       - Limited Inventory Awareness
       - Value of attending this call
       - Time-sensitive window

    **2. Motivation & Tailoring:**
       - **Motivation Checked?** (Yes/No)
       - **Identified Motivation:** (e.g., Rental Yield, Holiday Home).
       - **Tailored Pitch?** Did the CSM adapt the pitch? (Yes/No).

    **3. Objections (Bucketing):**
       Mark "Yes" if raised:
       - Price Related
       - Product Related
       - Location Related
       - ROI Related
       - Site Visit Related
       - Payment Terms Related

    **4. Q&A Log (Verbatim):**
       Format: "Cust: [Quote] -> CSM: [Quote]"

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
        
        # Data validation
        if len(parts) < 5:
            log_message(f"⚠️ {filename}: AI returned incomplete data.")
            return None
            
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
            "Closing/Urgency Tactic": parts[17],
            "File Name": filename
        }
    except Exception as e:
        log_message(f"❌ {filename}: AI Error - {str(e)}")
        return None

def generate_summaries(model, df):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    log_message("🧠 Generating Management Summaries...")
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
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(response.text.strip())
        return pd.DataFrame(data.get("CSM_Summaries", [])), pd.DataFrame(data.get("Team_Summary", []), columns=["Team Performance Insights"])
    except Exception as e:
        log_message(f"❌ Summary Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- UI LAYOUT ---
st.title("🛠️ G.O.A.A. Sales Auditor (Debug Mode)")

# Sidebar
with st.sidebar:
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_files = st.file_uploader("Upload Transcripts (PDF)", type=['pdf'], accept_multiple_files=True)
    start_btn = st.button("Start Processing")

# Main Content
status_area = st.empty()
progress_bar = st.progress(0)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis (Live)", "👨‍🏫 CSM Summary", "📈 Team Stats", "📝 Debug Logs"])

# Placeholders for live updates
with tab1: table_placeholder = st.empty()
with tab2: csm_placeholder = st.empty()
with tab3: team_placeholder = st.empty()
with tab4: log_placeholder = st.empty()

# --- PROCESSING LOGIC ---
if start_btn and uploaded_files and api_key:
    # Reset
    st.session_state.results_list = []
    st.session_state.logs = []
    
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        total = len(uploaded_files)
        log_message(f"🚀 Started processing {total} files...")
        
        for i, file in enumerate(uploaded_files):
            status_area.text(f"⏳ Processing {i+1}/{total}: {file.name}")
            
            # Extract
            text = extract_text_from_pdf(file)
            if not text:
                log_message(f"❌ {file.name}: Empty or unreadable PDF.")
                continue
                
            # Analyze
            res = analyze_single_call(model, text, file.name)
            if res:
                st.session_state.results_list.append(res)
                log_message(f"✅ {file.name}: Analyzed successfully.")
            
            # Update Live Table
            if st.session_state.results_list:
                table_placeholder.dataframe(pd.DataFrame(st.session_state.results_list))
                
            # Update Logs
            log_placeholder.text("\n".join(st.session_state.logs))
            
            # Progress
            progress_bar.progress((i + 1) / total)
            time.sleep(0.1)

        # Final Summaries
        status_area.success("✅ Analysis Complete! Generating Summaries...")
        df_final = pd.DataFrame(st.session_state.results_list)
        
        if not df_final.empty:
            df_csm, df_team = generate_summaries(model, df_final)
            csm_placeholder.dataframe(df_csm)
            team_placeholder.dataframe(df_team)
            
            # Excel Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_final.to_excel(writer, sheet_name='Analysis', index=False)
                df_csm.to_excel(writer, sheet_name='CSM Summary', index=False)
                df_team.to_excel(writer, sheet_name='Team Stats', index=False)
                
            st.download_button("📥 Download Final Report", output.getvalue(), "GOAA_Report.xlsx")
        else:
            st.error("No valid data was extracted. Check Debug Logs tab.")

    except Exception as e:
        st.error(f"Critical Error: {e}")
