# HIPAA Compliance Checklist — EKG Intelligence Platform

**Status:** Pre-development planning  
**Last updated:** 2026-04-13  
**Applies to:** Native mobile app (iOS/Android) with ECG analysis — from MVP through commercial product

---

## Product Tiers

This checklist covers two product stages. Both must be HIPAA-compliant, but the scope differs:

| Tier | Description | Users | Data Flow | Cloud |
|------|-------------|-------|-----------|-------|
| **Tier 1 — MVP** | On-device inference, single user, educational positioning | Individual consumers, students, paramedics | ECG stays on phone | None — fully offline |
| **Tier 2 — Commercial** | Clinic/practice sales, multi-user, doctor sharing, cloud sync | Clinics, practitioners, healthcare orgs | ECG may transit to cloud for sync/sharing/backup | Yes — HIPAA-covered backend required |

**Key insight:** Even Tier 1 handles PHI the moment a real patient's ECG is on screen. HIPAA applies to the app regardless of whether we call it "educational."

Items marked **[T1]** are required for MVP launch. Items marked **[T2]** are required before selling to clinics/practices.

---

## 1. Technical Safeguards (45 CFR 164.312)

### 1.1 Encryption

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 1.1.1 | Encryption at rest — all stored ECG data, results, and patient context encrypted using AES-256 | T1 | [ ] | Android: EncryptedSharedPreferences + SQLCipher. iOS: Data Protection (NSFileProtectionComplete) |
| 1.1.2 | Encryption keys stored in hardware-backed keystore | T1 | [ ] | Android Keystore / iOS Secure Enclave |
| 1.1.3 | No PHI written to application logs | T1 | [ ] | Strip all patient data from crash reports, analytics, debug logs |
| 1.1.4 | Encryption in transit — all network communication over TLS 1.2+ | T1 | [ ] | Applies to data exports (PDF share, email to doctor). No fallback to unencrypted channels |
| 1.1.5 | Temporary files (image processing, signal extraction) securely deleted after use | T1 | [ ] | Overwrite + delete, not just unlink |
| 1.1.6 | Cloud storage encrypted at rest (AES-256) | T2 | [ ] | If using GCS/AWS S3/Azure Blob — enable server-side encryption + customer-managed keys |
| 1.1.7 | End-to-end encryption for ECG data sync between devices | T2 | [ ] | Cloud backend stores only encrypted blobs it cannot decrypt |
| 1.1.8 | Database encryption for cloud backend | T2 | [ ] | Encrypted RDS/Cloud SQL + encrypted backups |

### 1.2 Access Controls

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 1.2.1 | App requires authentication to access stored ECG data | T1 | [ ] | Biometrics (FaceID/TouchID/fingerprint) or PIN |
| 1.2.2 | Auto-lock after inactivity timeout (configurable, default 5 min) | T1 | [ ] | Clear sensitive views from memory on background |
| 1.2.3 | No PHI accessible via app screenshots or recent-apps preview | T1 | [ ] | FLAG_SECURE (Android) / hidden content on app switch (iOS) |
| 1.2.4 | Unique user identification — no shared logins | T2 | [ ] | Each practitioner gets their own account, even at the same clinic |
| 1.2.5 | Role-based access control (RBAC) | T2 | [ ] | Admin (manage users, view audit logs), Practitioner (view/create ECGs), Read-only (view only) |
| 1.2.6 | Multi-factor authentication (MFA) for all accounts | T2 | [ ] | Required for admin accounts. Strongly recommended for all users |
| 1.2.7 | Automatic session expiration for cloud/web components | T2 | [ ] | Token-based auth with short-lived access tokens + refresh tokens |
| 1.2.8 | Emergency access procedure | T2 | [ ] | Break-glass process for when a practitioner is locked out but patient data is needed urgently |

### 1.3 Audit Controls

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 1.3.1 | Local audit log of PHI access events (view, export, delete) | T1 | [ ] | Encrypted log stored on-device |
| 1.3.2 | Audit log tamper-resistant (append-only, integrity hash) | T1 | [ ] | |
| 1.3.3 | User can view their own audit log | T1 | [ ] | Settings > Activity Log |
| 1.3.4 | Centralized audit log for cloud backend | T2 | [ ] | Who accessed which patient's ECG, when, from where |
| 1.3.5 | Admin can review all audit logs for their organization | T2 | [ ] | Required for clinic compliance officers |
| 1.3.6 | Audit logs retained for minimum 6 years | T2 | [ ] | HIPAA requires 6-year retention for compliance documentation |
| 1.3.7 | Audit log export capability | T2 | [ ] | For compliance audits and legal discovery |

### 1.4 Integrity Controls

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 1.4.1 | ECG data integrity verification (checksum on stored records) | T1 | [ ] | Detect corruption or tampering |
| 1.4.2 | Model file integrity check on app startup | T1 | [ ] | SHA-256 hash of ONNX/CoreML model embedded at build time |
| 1.4.3 | Cloud data integrity — checksums verified on upload/download | T2 | [ ] | Detect corruption in transit or at rest |
| 1.4.4 | Database backup integrity verification | T2 | [ ] | Automated backup testing on schedule |

---

## 2. Physical Safeguards (45 CFR 164.310)

### 2.1 Device-Level (Tier 1)

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 2.1.1 | App data wiped on device factory reset | [ ] | Standard OS behavior — verify no orphaned files |
| 2.1.2 | Remote wipe support (if device is lost) | [ ] | Leverage OS MDM capabilities, document in user guide |
| 2.1.3 | No PHI persisted on external/removable storage | [ ] | Block SD card writes on Android |
| 2.1.4 | App data excluded from unencrypted device backups | [ ] | Android: `android:allowBackup="false"`. iOS: exclude from iCloud unless encrypted |
| 2.1.5 | ECG images/signals never saved to public storage (Gallery/Photos) | [ ] | App-private directory only |

### 2.2 Infrastructure (Tier 2 — when cloud backend exists)

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 2.2.1 | Cloud provider physical security certification | [ ] | AWS/GCP/Azure all meet HIPAA physical safeguard requirements — verify and document |
| 2.2.2 | Development workstation security | [ ] | Encrypted hard drive, screen lock, no PHI on dev machines (use test data) |
| 2.2.3 | Device disposal protocol | [ ] | Secure wipe for test devices, old phones, retired servers |
| 2.2.4 | MFA on all infrastructure accounts (Google Cloud, AWS, etc.) | [ ] | A compromised admin account = full data breach |

---

## 3. Administrative Safeguards (45 CFR 164.308)

### 3.1 Organization (applies at both tiers)

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 3.1.1 | Designated Privacy Officer | T1 | [ ] | Solo developer = you are the Privacy Officer. Document it formally |
| 3.1.2 | Designated Security Officer | T1 | [ ] | Same person is fine. Document it |
| 3.1.3 | Risk Assessment documented | T1 | [ ] | Formal document identifying threats, vulnerabilities, likelihood, impact |
| 3.1.4 | Risk Management Plan | T1 | [ ] | Mitigation strategies for each identified risk |
| 3.1.5 | Incident Response Plan | T1 | [ ] | What happens if a vulnerability is found, breach notification procedures |
| 3.1.6 | Workforce training on HIPAA | T2 | [ ] | Required for every employee/contractor with PHI access. Document completion, repeat annually |
| 3.1.7 | Sanctions policy for workforce violations | T2 | [ ] | Written policy for what happens if an employee mishandles PHI |
| 3.1.8 | Termination procedure | T2 | [ ] | Revoke access immediately when employee/contractor leaves |

### 3.2 Business Associate Agreements (BAAs)

**A BAA is required with EVERY third party that may receive, store, process, or transmit PHI on your behalf.** This is one of the most commonly violated HIPAA requirements.

| # | Service | Tier | BAA Needed? | Status | Notes |
|---|---------|------|-------------|--------|-------|
| 3.2.1 | Cloud provider (AWS/GCP/Azure) | T2 | Yes | [ ] | All major providers offer HIPAA BAAs — must be signed before any PHI touches the cloud |
| 3.2.2 | Crash reporting (Sentry, Crashlytics, etc.) | T1 | Maybe | [ ] | Only if crash reports could contain PHI. Safest: strip all PHI from reports so no BAA needed |
| 3.2.3 | Analytics (Mixpanel, Firebase Analytics, etc.) | T1 | Maybe | [ ] | Only if analytics events reference PHI. Safest: collect zero PHI in analytics |
| 3.2.4 | Email service (SendGrid, SES, etc.) | T2 | Yes | [ ] | If sending results/reports to patients or doctors via email |
| 3.2.5 | Payment processor (Stripe, etc.) | T2 | No | [ ] | Payment data is PCI, not HIPAA — unless payment records are linked to PHI |
| 3.2.6 | Customer support platform (Zendesk, Intercom) | T2 | Yes | [ ] | If users send ECG screenshots or health details in support tickets |
| 3.2.7 | App distribution (App Store, Play Store) | T1 | No | [ ] | Apple/Google don't process your app's data |
| 3.2.8 | Google Workspace (Drive, Gmail) | T2 | Yes | [ ] | If using for business operations involving PHI. Google offers BAA — must opt in explicitly |
| 3.2.9 | Colab / training infrastructure | — | No | [ ] | Training uses public research datasets only, never real patient data. No BAA needed |

---

## 4. Breach Notification Rule (45 CFR 164.400-414)

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 4.1 | Breach notification procedure documented | T1 | [ ] | Individual notification within 60 days of discovery |
| 4.2 | HHS notification procedure | T1 | [ ] | 500+ individuals: notify HHS within 60 days. Fewer: annual log to HHS |
| 4.3 | State attorney general notification (if required) | T1 | [ ] | Varies by state — check applicable jurisdictions |
| 4.4 | Breach risk assessment methodology defined | T1 | [ ] | 4-factor test: nature of PHI, who accessed it, whether acquired/viewed, mitigation |
| 4.5 | Media notification for large breaches | T2 | [ ] | 500+ individuals in a single state: must notify prominent media outlets |

---

## 5. App-Specific Requirements

### 5.1 Data Handling

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 5.1.1 | Define minimum necessary PHI the app collects | T1 | [ ] | **De-identification strategy:** ECG signal + age + sex + symptoms. No names/SSN/insurance/DOB if possible. The less identifying info collected, the lower the risk |
| 5.1.2 | User can delete all their data (right to erasure) | T1 | [ ] | Single action: "Delete All My Data" in settings. Must also delete from cloud sync if T2 |
| 5.1.3 | User can export their data in portable format | T1 | [ ] | PDF report + raw signal file |
| 5.1.4 | Data retention policy defined and enforced | T1 | [ ] | Auto-delete after configurable period, or keep until manual delete |
| 5.1.5 | No PHI in push notifications | T1 | [ ] | Notifications say "New result ready" not "AFIB detected" |
| 5.1.6 | Patient consent workflow before storing/sharing ECG | T2 | [ ] | Document informed consent for data storage, sharing with other practitioners |
| 5.1.7 | Minimum necessary rule for data sharing | T2 | [ ] | When sharing ECG with another practitioner, share only what's needed — not the entire patient record |

### 5.2 Third-Party Dependencies

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 5.2.1 | Audit all third-party SDKs for data collection | T1 | [ ] | Firebase, analytics, crash reporters — each one is a potential BAA requirement |
| 5.2.2 | No ad SDKs | T1 | [ ] | Ads + PHI = automatic HIPAA violation |
| 5.2.3 | Analytics collect zero PHI | T1 | [ ] | Only: screen views, feature usage, app version, device model. Never ECG data or results |
| 5.2.4 | Crash reports stripped of PHI | T1 | [ ] | Custom crash handler that sanitizes stack traces |

### 5.3 Model & Inference

| # | Requirement | Tier | Status | Notes |
|---|-------------|------|--------|-------|
| 5.3.1 | Model runs on-device for Tier 1 — no server round-trip | T1 | [ ] | ONNX Runtime Mobile or CoreML |
| 5.3.2 | If cloud inference added (Tier 2): encrypted transit + no data retention on server | T2 | [ ] | Process ECG, return result, delete input immediately |
| 5.3.3 | No real patient ECG data sent for model improvement without explicit consent | T1 | [ ] | If federated learning is added later, requires separate consent + BAA |
| 5.3.4 | Model updates delivered via app store, not OTA sideload | T1 | [ ] | Ensures update integrity via platform signing |
| 5.3.5 | Clinical disclaimer displayed before every result | T1 | [ ] | "This is not a medical diagnosis. Consult a healthcare professional." |
| 5.3.6 | ECG images/signals never saved to public storage (Gallery/Photos) | T1 | [ ] | App-private directory only. Camera captures processed in memory, not saved to DCIM |
| 5.3.7 | App requests minimum necessary permissions | T1 | [ ] | Camera (for paper ECG capture) only. No contacts, location, microphone |

---

## 6. Multi-User & Clinic Features (Tier 2)

These requirements activate when the app is sold to clinics or practices with multiple users:

### 6.1 Account Management

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 6.1.1 | Unique user identification — no shared logins | [ ] | Each practitioner gets their own credentials |
| 6.1.2 | Organization/tenant isolation | [ ] | Clinic A cannot see Clinic B's data, even if on same backend |
| 6.1.3 | Admin can provision/deprovision users | [ ] | Adding new staff, removing departed staff |
| 6.1.4 | Access revocation takes effect immediately | [ ] | Deprovisioned user's active sessions terminated within minutes |
| 6.1.5 | Password complexity and rotation policy | [ ] | Or better: passwordless auth (biometrics + device trust) |

### 6.2 Data Sharing Between Practitioners

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 6.2.1 | Share ECG results with specific practitioners only (not broadcast) | [ ] | Explicit recipient selection |
| 6.2.2 | Sharing creates an audit trail entry | [ ] | Who shared what with whom, when |
| 6.2.3 | Recipient cannot re-share without explicit permission | [ ] | Prevent uncontrolled data propagation |
| 6.2.4 | Revoke sharing access retroactively | [ ] | Doctor leaves practice — their access to shared ECGs must be revocable |

### 6.3 Cloud Backend Requirements

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 6.3.1 | HIPAA-eligible cloud service with signed BAA | [ ] | AWS, GCP, or Azure — all offer HIPAA BAAs |
| 6.3.2 | Data residency — PHI stored in jurisdiction-appropriate region | [ ] | US data stays in US. EU data may need EU region (GDPR overlap) |
| 6.3.3 | Regular penetration testing | [ ] | At least annual. More frequent for critical changes |
| 6.3.4 | Vulnerability scanning automated | [ ] | CI/CD pipeline includes dependency and container scanning |
| 6.3.5 | Intrusion detection / monitoring | [ ] | Alert on unusual access patterns, bulk downloads, etc. |
| 6.3.6 | Backup and disaster recovery plan | [ ] | Encrypted backups, tested restore procedure, documented RTO/RPO |

---

## 7. FDA / Regulatory Status

**Decision: FDA 510(k) is deferred indefinitely.** Solo developer, limited resources. The cost ($50K-$200K clinical study + 12-18 months) is not viable at this stage.

### What this means for the app

- The app **cannot claim to diagnose** medical conditions. It must be positioned as an **educational/informational tool**, not a medical device.
- Required disclaimer on every result screen: *"For educational purposes only. This is not a medical diagnosis. Not FDA-cleared. Always consult a qualified healthcare professional."*
- App store listing must avoid medical device language ("detects", "diagnoses"). Use: "analyzes", "identifies patterns", "educational tool".
- This is the same approach used by many ECG apps pre-clearance (including early AliveCor, Kardia).

### Revisit FDA when

- The app has traction (users, revenue, or investor interest)
- A clinical partner offers to co-fund a validation study
- Regulatory pathway becomes necessary for hospital/clinic distribution

### Items to keep in mind even without FDA

| # | Requirement | Notes |
|---|-------------|-------|
| 7.1 | Clinical disclaimer on every result | Legal protection — non-negotiable |
| 7.2 | No "diagnose" language in marketing or UI | App store rejection risk + legal liability |
| 7.3 | Keep training/validation records clean | If FDA is pursued later, having good records from the start saves months |
| 7.4 | Document model accuracy per class | Transparency builds trust even without formal clearance |

---

## 8. Documentation Required

### Tier 1 (before MVP launch)

| Document | Purpose | Status |
|----------|---------|--------|
| Privacy Impact Assessment (PIA) | What PHI is collected, why, how it's protected | [ ] |
| Data Flow Diagram | Visual map: ECG input -> processing -> storage -> display -> export | [ ] |
| Risk Assessment | Threat model specific to on-device medical data | [ ] |
| Incident Response Plan | Step-by-step breach handling procedure | [ ] |
| Privacy Policy (user-facing) | Plain-language explanation of data practices for app store listing | [ ] |
| Terms of Service | Legal disclaimers, educational-use positioning, liability limitations | [ ] |

### Tier 2 (before commercial/clinic sales)

| Document | Purpose | Status |
|----------|---------|--------|
| System Security Plan | Full architecture with data flows, encryption boundaries, access points | [ ] |
| BAA Template | Ready for every third-party service that touches PHI | [ ] |
| Business Continuity Plan | What happens if the backend goes down, data recovery procedures | [ ] |
| Employee Onboarding Checklist | HIPAA training, access provisioning, NDA for every team member | [ ] |
| Vendor Assessment Checklist | Due diligence template for evaluating new third-party services | [ ] |
| Compliance Audit Schedule | Annual HIPAA self-audit + documentation review | [ ] |

---

## Summary & Next Steps

### The two-tier approach

**Tier 1 (MVP)** is achievable as a solo developer. Offline-first, on-device inference, no cloud backend. HIPAA requirements are mostly one-time implementation choices baked into the app architecture. The documentation is 4-6 short documents you can write in a weekend.

**Tier 2 (commercial)** requires a cloud backend, multi-user auth, BAAs, and significantly more operational overhead. This is when you need either funding, a co-founder, or a managed HIPAA-compliant backend service (like AWS for Healthcare, Aptible, or Datica) that handles the infrastructure compliance for you.

### Immediate next steps (Tier 1 — solo developer)

1. **Privacy Impact Assessment** — 2-3 pages defining what data the app collects and why.
2. **Data Flow Diagram** — One page: ECG input -> preprocessing -> model inference -> result display -> local encrypted storage -> export/delete.
3. **Risk Assessment** — Threats: device theft, malware, screenshot capture, backup extraction, shoulder surfing.
4. **Privacy Policy + Terms of Service** — Required for app store submission. Can use templates but must be accurate.
5. **Build with security from line one** — SQLCipher, no PHI in logs, biometric gate, FLAG_SECURE.

### When to start Tier 2 planning

- When you have paying users asking for multi-user / clinic features
- When a clinic or practice expresses intent to purchase
- When you raise funding or bring on a technical co-founder
- **Not before** — premature cloud infrastructure is wasted money and complexity

### What runs in parallel now

- V3.2 Colab training (in progress)
- Model export to ONNX
- App UI prototyping with synthetic ECG data
- React Native scaffold with encrypted storage from day one
- Tier 1 documentation (PIA, risk assessment, data flow diagram)

### Reality check

- **Tier 1 HIPAA compliance:** manageable solo. Mostly architectural decisions, not ongoing overhead.
- **Tier 2 HIPAA compliance:** real operational cost. Budget $500-2000/month for HIPAA-compliant infrastructure (Aptible, AWS HIPAA-eligible services, or similar).
- **FDA: deferred.** Position as educational/informational. Disclaimers everywhere.
- **Biggest unlock for going commercial:** a managed HIPAA backend (Aptible, AWS HealthLake, etc.) that handles infrastructure compliance so you can focus on the app.
