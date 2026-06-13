# Competitor Analysis: EKG Digitization and Analysis
## Focused on Mobile Paper Strip Scanning & AI Interpretation

This analysis evaluates the leading players in the EKG digitization space, with a specific focus on the technical hurdles of converting paper-based strips into diagnostic data using smartphone cameras.

---

### 1. Competitive Overview Matrix

| Feature | PMcardio | ECG Buddy | Cardiomatics | Anumana |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Platform** | Mobile App (iOS/Android) | Mobile App (Real-time) | Cloud / Web-based | Clinical AI Integration |
| **Digitization Tech** | Perspective-Warping CV | Video-stream processing | Batch Cloud Processing | Data-lake AI models |
| **Key Advantage** | High-end Clinical UX | Speed / Instant Analysis | Reporting Depth | Predictive Diagnostics |
| **Target Audience** | General Practitioners | Emergency Medicine | Research / Holter Labs | Specialist Cardiology |

---

### 2. Deep Dives

#### **PMcardio (Powerful Medical)**
* **The Approach:** Uses advanced computer vision to perform "Perspective Rectification." It detects the four corners of an EKG grid and flattens the image to eliminate lens distortion.
* **Relevance to YAIR:** They handle the **inversion** problem via OCR (identifying Lead labels like II, aVR, etc.). If your pipeline is failing on FX-8200 reference strips, PMcardio's logic of "Identify Lead -> Orient Signal" is the gold standard.
* **Strengths:** CE Class IIb certified; very clean UI for "Yielding Reports."

#### **ECG Buddy**
* **The Approach:** Optimized for the "Point and Shoot" workflow. It often uses the smartphone's video feed to find the frame with the least motion blur and best lighting.
* **Relevance to YAIR:** It demonstrates how to handle **hand-tremor noise** and variable lighting—critical for your phone-camera-based web app.
* **Strengths:** Extremely fast; designed for acute care settings where every second counts.

#### **Cardiomatics**
* **The Approach:** A cloud-first approach. They focus on the signal-to-noise ratio (SNR) and filtering out the 50/60Hz "banding" noise introduced by phone cameras or indoor lighting.
* **Relevance to YAIR:** Their backend can handle long-form data (Holters) but provides a high-fidelity bridge for paper records.
* **Strengths:** Highly reliable diagnostic accuracy; strong presence in research environments.

#### **Anumana**
* **The Approach:** Focuses on "Hidden Biomarkers." They leverage the Mayo Clinic's database to detect conditions that humans (and basic CNNs) might miss, such as Low Ejection Fraction (EF).
* **Relevance to YAIR:** This represents the "Future State" of your EKG model—moving from arrhythmia identification to predictive health insights.
* **Strengths:** Deep clinical data backing; predictive capabilities for future heart failure.

---

### 3. Technical Implications for EKGMobile & YAIR

Based on the current digitization challenges (inversion, variable $fs$, and camera distortion), the following strategies are recommended:

1.  **Perspective Correction:** Like PMcardio, the pipeline must "warp" the paper to a flat rectangle before signal extraction.
2.  **Auto-Polarity Check:** Implement a logic check (e.g., `abs(min) > abs(max)`) to detect and flip inverted leads (like aVR) before NeuroKit2 processing.
3.  **Local Grid Calibration:** Instead of a global $fs$, calculate pixels-per-mm locally to account for lens distortion (pincushion/barrel effects).
4.  **Signal Resampling:** Standardize all camera-extracted signals to a fixed $fs$ (e.g., 500Hz) using spline interpolation to ensure interval accuracy (PR, QRS, QTc).

---
*Prepared for the EKG/YAIR Project Workflow Architecture*