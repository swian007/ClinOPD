import os
import json
import sqlite3
from datetime import datetime
import streamlit as st
from PIL import Image
from openai import OpenAI

# ------------------------------------------------------------------------------
# 1. DATABASE & STORAGE INITIALIZATION
# ------------------------------------------------------------------------------
DB_FILE = "clinical_records.db"
UPLOADS_DIR = "opd_slips"

if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visit_date TEXT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            address TEXT,
            chief_complaint TEXT,
            duration TEXT,
            onset TEXT,
            pain_details TEXT,
            previous_meds TEXT,
            ai_questions TEXT,
            final_diagnosis TEXT,
            doctor_management TEXT,
            ai_comparison TEXT,
            opd_slip_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------------------
# 2. AI CLINICAL ENGINE (OPENAI INTEGRATION)
# ------------------------------------------------------------------------------
def get_ai_client(api_key):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def analyze_symptoms_and_generate_questions(client, patient_data):
    """
    Analyzes vague or structured patient complaints and provides diagnostic follow-up questions.
    """
    prompt = f"""
    You are an expert clinical diagnostic assistant tailored for outpatient clinic setups in South Asia/Pakistan.
    A patient has presented with the following intake details:
    - Age/Gender: {patient_data['age']} years old, {patient_data['gender']}
    - Location/Address: {patient_data['address']}
    - Chief Complaint: {patient_data['chief_complaint']}
    - Duration: {patient_data['duration']}
    - Onset: {patient_data['onset']}
    - Pain Details: {patient_data['pain_details']}
    - Previous Medications Used: {patient_data['previous_meds']}

    Instructions:
    1. Patient complaints in this demographic are often vague (e.g., general fatigue, generalized body ache, epigastric distress, dizziness).
    2. Analyze the input data and generate 4-6 high-yield, specific follow-up questions that the clinician should ask next to narrow down the differential diagnosis effectively.
    3. Provide possible differential diagnoses to keep in mind.

    Format your output cleanly in Markdown with bullet points. Language must be professional English.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior physician assisting an OPD practitioner with clinical reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

def compare_management_plan(client, patient_data, diagnosis, doctor_plan):
    """
    Compares physician's entered management plan with evidence-based guideline recommendations.
    """
    prompt = f"""
    You are an expert clinical pharmacologist and medical supervisor auditing outpatient care plans.

    PATIENT CONTEXT:
    - Age/Gender: {patient_data['age']}, {patient_data['gender']}
    - Address/Location: {patient_data['address']}
    - Chief Complaint: {patient_data['chief_complaint']}
    - Previous Meds: {patient_data['previous_meds']}
    - Working/Final Diagnosis: {diagnosis}

    DOCTOR'S ENTERED MANAGEMENT PLAN:
    {doctor_plan}

    INSTRUCTIONS:
    Provide a side-by-side comparative analysis of the doctor's management plan against standard clinical practice guidelines (e.g., WHO, NICE, local protocols).
    Structure your output as follows:
    1. **Guideline-Recommended Management**: What is standardly recommended for this diagnosis.
    2. **Side-by-Side Assessment**:
       - **Agreements / Strengths**: What the doctor prescribed correctly.
       - **Gaps / Omissions**: Standard drugs or non-pharmacological advice missed (e.g., lifestyle modifications, baseline lab work).
       - **Potential Risks / Interactions**: Drug interactions, contraindications, or overuse risks (e.g., unnecessary antibiotics, steroid abuse, NSAID toxicity).
    3. **Optimized Treatment Recommendation**: A clean, updated prescription protocol.

    Keep the response fully in professional English.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a clinical pharmacologist auditing prescription accuracy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

# ------------------------------------------------------------------------------
# 3. STREAMLIT FRONTEND USER INTERFACE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Intelligent OPD Clinical Assistant", layout="wide")

# Sidebar Configuration
st.sidebar.title("Clinical Assistant Setup")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

if not api_key_input:
    st.sidebar.warning("Please enter your OpenAI API key above to enable dynamic AI clinical analysis.")

ai_client = get_ai_client(api_key_input) if api_key_input else None

menu = st.sidebar.radio("Navigation", ["New Consultation Engine", "Patient Database & Search"])

# ------------------------------------------------------------------------------
# TAB 1: NEW CONSULTATION ENGINE
# ------------------------------------------------------------------------------
if menu == "New Consultation Engine":
    st.title("Intelligent OPD Consultation Engine")
    st.caption("Structured Intake, AI Diagnostic Clarification, and Prescription Auditing")

    st.markdown("---")
    
    # Section 1: Demographics
    st.subheader("1. Patient Demographics")
    col1, col2, col3, col4 = st.columns([3, 1, 1, 3])
    with col1:
        name = st.text_input("Patient Name")
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col4:
        address = st.text_input("Location / Address")

    # Section 2: Clinical Intake
    st.subheader("2. Symptom Intake & History of Present Illness")
    chief_complaint = st.text_area(
        "Chief Complaint", 
        placeholder="Enter symptoms (e.g., generalized body weakness, epigastric burning, dizziness, fever)..."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        duration = st.text_input("Duration", placeholder="e.g., 3 days, 2 weeks")
        onset = st.selectbox("Onset Pattern", ["Gradual", "Sudden", "Intermittent", "Not Specified"])
    with col_b:
        is_pain = st.checkbox("Is Pain a presenting feature?")
        pain_details = "N/A"
        if is_pain:
            p_site = st.text_input("Site & Radiation")
            p_char = st.text_input("Character & Severity (1-10)")
            p_fact = st.text_input("Aggravating / Relieving Factors")
            pain_details = f"Site/Radiation: {p_site} | Character/Severity: {p_char} | Factors: {p_fact}"

    previous_meds = st.text_area("Previous / Current Medications Used", placeholder="e.g., Self-medicated with OTC painkillers, antibiotics...")

    # Dynamic Analysis Trigger
    ai_questions_output = ""
    if st.button("Analyze Symptoms & Generate Diagnostic Questions"):
        if not chief_complaint:
            st.error("Please enter a chief complaint to proceed with analysis.")
        elif not ai_client:
            st.error("API Key missing. Please provide a valid OpenAI API key in the sidebar.")
        else:
            with st.spinner("Analyzing symptoms and generating diagnostic prompts..."):
                patient_data = {
                    "age": age, "gender": gender, "address": address,
                    "chief_complaint": chief_complaint, "duration": duration,
                    "onset": onset, "pain_details": pain_details,
                    "previous_meds": previous_meds
                }
                ai_questions_output = analyze_symptoms_and_generate_questions(ai_client, patient_data)
                st.session_state['ai_questions'] = ai_questions_output

    if 'ai_questions' in st.session_state and st.session_state['ai_questions']:
        st.markdown("### Suggested Clinical Questions & Differential Considerations")
        st.info(st.session_state['ai_questions'])

    st.markdown("---")
    
    # Section 3: Final Diagnosis & Plan Comparison
    st.subheader("3. Diagnosis & Management Audit")
    
    final_diagnosis = st.text_input("Working / Final Diagnosis", placeholder="e.g., Acute Gastroenteritis, Essential Hypertension, GERD")
    doctor_management = st.text_area(
        "Enter Prescribed Management Plan", 
        placeholder="List medications, dosages, duration, and advice given (e.g., Tab Ciprofloxacin 500mg BD x 5 days, ORS)...",
        height=120
    )

    ai_comparison_output = ""
    if st.button("Compare Management Plan with AI Guidelines"):
        if not final_diagnosis or not doctor_management:
            st.error("Please enter both the Final Diagnosis and Doctor's Management Plan.")
        elif not ai_client:
            st.error("API Key missing. Please provide a valid OpenAI API key in the sidebar.")
        else:
            with st.spinner("Performing comparative audit against clinical practice guidelines..."):
                patient_data = {
                    "age": age, "gender": gender, "address": address,
                    "chief_complaint": chief_complaint, "previous_meds": previous_meds
                }
                ai_comparison_output = compare_management_plan(ai_client, patient_data, final_diagnosis, doctor_management)
                st.session_state['ai_comparison'] = ai_comparison_output

    if 'ai_comparison' in st.session_state and st.session_state['ai_comparison']:
        st.markdown("### Clinical Comparison & Audit")
        st.success(st.session_state['ai_comparison'])

    st.markdown("---")
    
    # Section 4: Attachments & Save Record
    st.subheader("4. OPD Slip Upload & Record Archival")
    uploaded_file = st.file_uploader("Attach Photo of OPD Slip / Lab Reports", type=["jpg", "png", "jpeg"])

    if st.button("Save Complete Patient Record"):
        if not name or not chief_complaint:
            st.error("Patient Name and Chief Complaint are required to save the record.")
        else:
            image_path = ""
            if uploaded_file is not None:
                file_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
                image_path = os.path.join(UPLOADS_DIR, file_filename)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO patients (
                    visit_date, name, age, gender, address, chief_complaint,
                    duration, onset, pain_details, previous_meds,
                    ai_questions, final_diagnosis, doctor_management,
                    ai_comparison, opd_slip_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M'), name, age, gender, address, chief_complaint,
                duration, onset, pain_details, previous_meds,
                st.session_state.get('ai_questions', ''), final_diagnosis, doctor_management,
                st.session_state.get('ai_comparison', ''), image_path
            ))
            conn.commit()
            conn.close()
            
            st.balloons()
            st.success(f"Patient record for '{name}' successfully committed to database!")
            
            # Clear temporary state
            if 'ai_questions' in st.session_state: del st.session_state['ai_questions']
            if 'ai_comparison' in st.session_state: del st.session_state['ai_comparison']

# ------------------------------------------------------------------------------
# TAB 2: PATIENT DATABASE & SEARCH
# ------------------------------------------------------------------------------
elif menu == "Patient Database & Search":
    st.title("Patient Database & Historical Records")
    
    search_query = st.text_input("Search records by Patient Name or Location Address", "")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    if search_query:
        c.execute("SELECT * FROM patients WHERE name LIKE ? OR address LIKE ? ORDER BY id DESC", 
                  (f'%{search_query}%', f'%{search_query}%'))
    else:
        c.execute("SELECT * FROM patients ORDER BY id DESC LIMIT 20")

    records = c.fetchall()
    conn.close()

    st.write(f"Displaying **{len(records)}** patient records:")

    for row in records:
        with st.expander(f"ID #{row[0]} | {row[2]} ({row[3]} yrs, {row[4]}) - Location: {row[5]} | Date: {row[1]}"):
            col_l, col_r = st.columns(2)
            with col_l:
                st.write(f"**Chief Complaint:** {row[6]}")
                st.write(f"**Duration / Onset:** {row[7]} / {row[8]}")
                st.write(f"**Pain Profile:** {row[9]}")
                st.write(f"**Previous Medications:** {row[10]}")
            with col_r:
                st.write(f"**Working Diagnosis:** {row[12]}")
                st.write(f"**Prescribed Management Plan:** {row[13]}")
            
            st.markdown("---")
            if row[11]:
                st.markdown("**AI Diagnostic Questions:**")
                st.caption(row[11])
            if row[14]:
                st.markdown("**AI Management Plan Audit:**")
                st.info(row[14])
            if row[15] and os.path.exists(row[15]):
                st.markdown("**Attached OPD Slip:**")
                image = Image.open(row[15])
                st.image(image, width=400)
