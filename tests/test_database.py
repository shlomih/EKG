"""Tests for database_setup.py — run with: python -m pytest tests/test_database.py -v"""
import os
import sqlite3
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import database_setup as db


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect all DB calls to a temp file so tests never touch ekg_platform.db."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", db_path)
    yield db_path


# ── init_db ──────────────────────────────────────────────────────────────────

def test_init_db_creates_tables(tmp_db):
    db.init_db()
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"patients", "ekg_records", "analysis_results", "consultations"}.issubset(tables)


def test_init_db_idempotent(tmp_db):
    db.init_db()
    db.init_db()  # should not raise


def test_init_db_migrates_old_schema(tmp_db):
    """Old schema with only 'id' column should be dropped and recreated cleanly."""
    conn = sqlite3.connect(tmp_db)
    conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    db.init_db()  # should detect incompatible schema and recreate

    conn = sqlite3.connect(tmp_db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(patients)").fetchall()}
    conn.close()
    assert "first_name" in cols
    assert "last_name" in cols
    assert "id_number" in cols


# ── save_patient ──────────────────────────────────────────────────────────────

def test_save_patient_returns_id(tmp_db):
    db.init_db()
    pid = db.save_patient("Alice", "Smith", "ID001", 45, "F")
    assert isinstance(pid, str)
    assert len(pid) > 0


def test_save_patient_new_patient(tmp_db):
    db.init_db()
    pid = db.save_patient("Bob", "Jones", "ID002", 60, "M", k_level=4.2)
    patient = db.get_patient(patient_id=pid)
    assert patient["first_name"] == "Bob"
    assert patient["last_name"] == "Jones"
    assert patient["id_number"] == "ID002"
    assert patient["age"] == 60
    assert patient["sex"] == "M"


def test_save_patient_updates_existing(tmp_db):
    db.init_db()
    pid1 = db.save_patient("Carol", "White", "ID003", 30, "F")
    pid2 = db.save_patient("Carol", "White-Updated", "ID003", 31, "F")
    assert pid1 == pid2  # same patient
    patient = db.get_patient(patient_id=pid1)
    assert patient["last_name"] == "White-Updated"
    assert patient["age"] == 31


def test_save_patient_no_id_number(tmp_db):
    db.init_db()
    pid = db.save_patient("Dave", "Brown", "", 50, "M")
    assert isinstance(pid, str)


def test_save_patient_clinical_flags(tmp_db):
    db.init_db()
    pid = db.save_patient("Eve", "Green", "ID005", 35, "F",
                          has_pacemaker=True, is_athlete=False, is_pregnant=True, k_level=3.8)
    p = db.get_patient(patient_id=pid)
    assert p["has_pacemaker_icd"] == 1
    assert p["is_pregnant"] == 1
    assert abs(p["k_level"] - 3.8) < 0.01


# ── get_patient ───────────────────────────────────────────────────────────────

def test_get_patient_by_id_number(tmp_db):
    db.init_db()
    db.save_patient("Frank", "Lee", "ID006", 55, "M")
    p = db.get_patient(id_number="ID006")
    assert p is not None
    assert p["first_name"] == "Frank"


def test_get_patient_not_found(tmp_db):
    db.init_db()
    assert db.get_patient(patient_id="nonexistent") is None


def test_get_patient_no_args(tmp_db):
    db.init_db()
    assert db.get_patient() is None


# ── list_patients ─────────────────────────────────────────────────────────────

def test_list_patients_empty(tmp_db):
    db.init_db()
    assert db.list_patients() == []


def test_list_patients_returns_all(tmp_db):
    db.init_db()
    db.save_patient("G", "H", "ID007", 40, "M")
    db.save_patient("I", "J", "ID008", 41, "F")
    patients = db.list_patients()
    assert len(patients) == 2


# ── save_ekg_record ───────────────────────────────────────────────────────────

def test_save_ekg_record(tmp_db):
    db.init_db()
    pid = db.save_patient("Kay", "L", "ID009", 65, "F")
    eid = db.save_ekg_record(pid, acquisition_source="scan", ai_model_version="v3.2b")
    assert isinstance(eid, str)
    records = db.get_patient_records(pid)
    assert len(records) == 1
    assert records[0]["acquisition_source"] == "scan"


# ── save_analysis ─────────────────────────────────────────────────────────────

def test_save_analysis(tmp_db):
    db.init_db()
    pid = db.save_patient("Mel", "N", "ID010", 70, "M")
    eid = db.save_ekg_record(pid)
    aid = db.save_analysis(eid, classification="AFIB", confidence=0.92,
                           probabilities={"AFIB": 0.92, "NORM": 0.05},
                           heart_rate=88.0, urgency="urgent")
    assert isinstance(aid, str)
    records = db.get_patient_records(pid)
    assert records[0]["classification"] == "AFIB"
    assert records[0]["urgency"] == "urgent"
