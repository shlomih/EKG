"""
database_setup.py
=================
SQLite database for the EKG Intelligence Platform.

Stores patients (with name + ID for differentiation), EKG records,
analysis results, and consultation requests.

Usage:
    from database_setup import (
        init_db, save_patient, get_patient, list_patients,
        save_ekg_record, get_patient_records, save_analysis,
    )
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = "ekg_platform.db"


def get_connection():
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Patient Table -- name + ID for differentiation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            id_number TEXT UNIQUE,
            age INTEGER,
            sex TEXT DEFAULT 'M',
            has_pacemaker_icd BOOLEAN DEFAULT 0,
            is_athlete BOOLEAN DEFAULT 0,
            is_pregnant BOOLEAN DEFAULT 0,
            k_level FLOAT DEFAULT 4.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. EKG Record Table (Provenance)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ekg_records (
            ekg_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            acquisition_source TEXT,
            lead_count INTEGER DEFAULT 12,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ai_model_version TEXT,
            notes TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # 3. Analysis Results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            analysis_id TEXT PRIMARY KEY,
            ekg_id TEXT NOT NULL,
            classification TEXT,
            confidence FLOAT,
            probabilities TEXT,
            heart_rate FLOAT,
            pr_interval FLOAT,
            qrs_duration FLOAT,
            qtc FLOAT,
            st_summary TEXT,
            urgency TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(ekg_id) REFERENCES ekg_records(ekg_id)
        )
    ''')

    # 4. Consultations
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
    print("Database initialized (ekg_platform.db)")


# -------------------------------------------------------------
# Patient CRUD
# -------------------------------------------------------------

def save_patient(first_name, last_name, id_number, age, sex,
                 has_pacemaker=False, is_athlete=False, is_pregnant=False,
                 k_level=4.0):
    """Save or update a patient. Returns patient_id."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if patient exists by id_number
    if id_number:
        cursor.execute("SELECT patient_id FROM patients WHERE id_number = ?", (id_number,))
        row = cursor.fetchone()
        if row:
            # Update existing patient
            cursor.execute('''
                UPDATE patients SET
                    first_name=?, last_name=?, age=?, sex=?,
                    has_pacemaker_icd=?, is_athlete=?, is_pregnant=?,
                    k_level=?, updated_at=?
                WHERE id_number=?
            ''', (first_name, last_name, age, sex,
                  has_pacemaker, is_athlete, is_pregnant,
                  k_level, datetime.now().isoformat(), id_number))
            conn.commit()
            pid = row["patient_id"]
            conn.close()
            return pid

    # New patient
    patient_id = str(uuid.uuid4())[:8]
    cursor.execute('''
        INSERT INTO patients (patient_id, first_name, last_name, id_number,
                              age, sex, has_pacemaker_icd, is_athlete,
                              is_pregnant, k_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, first_name, last_name, id_number,
          age, sex, has_pacemaker, is_athlete, is_pregnant, k_level))
    conn.commit()
    conn.close()
    return patient_id


def get_patient(patient_id=None, id_number=None):
    """Look up a patient by patient_id or id_number."""
    conn = get_connection()
    cursor = conn.cursor()
    if patient_id:
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    elif id_number:
        cursor.execute("SELECT * FROM patients WHERE id_number = ?", (id_number,))
    else:
        conn.close()
        return None
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_patients():
    """Return all patients ordered by last update."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------
# EKG Record CRUD
# -------------------------------------------------------------

def save_ekg_record(patient_id, acquisition_source="dataset", lead_count=12,
                    ai_model_version=None, notes=None):
    """Save a new EKG record. Returns ekg_id."""
    conn = get_connection()
    ekg_id = str(uuid.uuid4())[:8]
    conn.execute('''
        INSERT INTO ekg_records (ekg_id, patient_id, acquisition_source,
                                 lead_count, ai_model_version, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ekg_id, patient_id, acquisition_source, lead_count,
          ai_model_version, notes))
    conn.commit()
    conn.close()
    return ekg_id


def get_patient_records(patient_id):
    """Get all EKG records for a patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT e.*, a.classification, a.confidence, a.heart_rate, a.urgency
        FROM ekg_records e
        LEFT JOIN analysis_results a ON e.ekg_id = a.ekg_id
        WHERE e.patient_id = ?
        ORDER BY e.captured_at DESC
    ''', (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------
# Analysis Results
# -------------------------------------------------------------

def save_analysis(ekg_id, classification=None, confidence=None,
                  probabilities=None, heart_rate=None, pr_interval=None,
                  qrs_duration=None, qtc=None, st_summary=None, urgency=None):
    """Save analysis results for an EKG record."""
    import json
    conn = get_connection()
    analysis_id = str(uuid.uuid4())[:8]
    prob_str = json.dumps(probabilities) if probabilities else None
    conn.execute('''
        INSERT INTO analysis_results (analysis_id, ekg_id, classification,
                                       confidence, probabilities, heart_rate,
                                       pr_interval, qrs_duration, qtc,
                                       st_summary, urgency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (analysis_id, ekg_id, classification, confidence, prob_str,
          heart_rate, pr_interval, qrs_duration, qtc, st_summary, urgency))
    conn.commit()
    conn.close()
    return analysis_id


if __name__ == "__main__":
    init_db()

