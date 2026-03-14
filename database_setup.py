import sqlite3

def init_db():
    conn = sqlite3.connect('ekg_platform.db')
    cursor = conn.cursor()

    # 1. Patient Table (with Logic Inverters)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            age INTEGER,
            sex INTEGER,
            has_pacemaker_icd BOOLEAN DEFAULT 0,
            is_athlete BOOLEAN DEFAULT 0,
            is_pregnant BOOLEAN DEFAULT 0,
            k_level FLOAT DEFAULT 4.0
        )
    ''')

    # 2. EKG Record Table (Provenance)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ekg_records (
            ekg_id TEXT PRIMARY KEY,
            patient_id TEXT,
            acquisition_source TEXT,
            lead_count INTEGER,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ai_model_version TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # 3. Consultations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ekg_id TEXT,
            status TEXT DEFAULT 'Pending',
            human_findings TEXT,
            reviewed_at TIMESTAMP,
            FOREIGN KEY(ekg_id) REFERENCES ekg_records(ekg_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✓ Clinical Database Initialized (ekg_platform.db)")

if __name__ == "__main__":
    init_db()
    
