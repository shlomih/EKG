/**
 * ReportTemplate — HTML template for ECG analysis reports.
 *
 * Generates clinical report HTML with patient info, ECG conditions, and urgency guidance.
 * Suitable for PDF export via react-native-html-to-pdf.
 */

import { AnalysisResult } from '../types/Analysis';
import { Patient } from '../types/Patient';

/**
 * Generate HTML report from analysis data.
 */
export function generateReportHTML(
  patient: Patient,
  analysis: AnalysisResult,
  conditions: Array<{ name: string; urgency: string }>,
): string {
  const timestamp = new Date(analysis.created_at).toLocaleString();
  const urgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical':
        return '#FF4444';
      case 'high':
        return '#FFB800';
      case 'normal':
      default:
        return '#00E5B0';
    }
  };

  const conditionRows = conditions
    .map(
      (cond) => `
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="padding: 10px; color: #333;">${cond.name}</td>
      <td style="padding: 10px; color: ${urgencyColor(cond.urgency)}; font-weight: bold;">
        ${cond.urgency.toUpperCase()}
      </td>
    </tr>
  `,
    )
    .join('');

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      background: white;
    }
    .container { max-width: 850px; margin: 0 auto; padding: 40px 20px; }
    .header {
      text-align: center;
      border-bottom: 3px solid #00E5B0;
      padding-bottom: 20px;
      margin-bottom: 30px;
    }
    .header h1 { color: #071312; font-size: 28px; margin-bottom: 4px; }
    .header p { color: #666; font-size: 14px; }
    .section {
      margin-bottom: 30px;
      page-break-inside: avoid;
    }
    .section h2 {
      color: #00E5B0;
      font-size: 18px;
      border-bottom: 1px solid #ddd;
      padding-bottom: 8px;
      margin-bottom: 12px;
    }
    .info-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin-bottom: 15px;
    }
    .info-item {
      background: #f9f9f9;
      padding: 12px;
      border-radius: 6px;
      border-left: 3px solid #00E5B0;
    }
    .info-label {
      color: #666;
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .info-value {
      color: #071312;
      font-size: 16px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 15px;
    }
    th {
      background: #f0f0f0;
      padding: 10px;
      text-align: left;
      font-weight: 600;
      color: #333;
      border-bottom: 2px solid #ddd;
    }
    .critical {
      color: #FF4444;
      font-weight: bold;
    }
    .high {
      color: #FFB800;
      font-weight: bold;
    }
    .normal {
      color: #00E5B0;
      font-weight: bold;
    }
    .disclaimer {
      background: #fff9e6;
      border-left: 4px solid #FF6B6B;
      padding: 15px;
      margin-top: 30px;
      font-size: 12px;
      color: #666;
      border-radius: 4px;
      page-break-inside: avoid;
    }
    .footer {
      text-align: center;
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid #ddd;
      font-size: 11px;
      color: #999;
    }
    @media print {
      body { background: white; }
      .section { page-break-inside: avoid; }
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <h1>ECG Analysis Report</h1>
      <p>EKG Intelligence Platform — V3 Multilabel</p>
    </div>

    <!-- Patient Information -->
    <div class="section">
      <h2>Patient Information</h2>
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">Name</div>
          <div class="info-value">${patient.first_name} ${patient.last_name}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Age</div>
          <div class="info-value">${patient.age ?? 'N/A'} years</div>
        </div>
        <div class="info-item">
          <div class="info-label">Sex</div>
          <div class="info-value">${patient.sex === 'M' ? 'Male' : patient.sex === 'F' ? 'Female' : 'Other'}</div>
        </div>
        <div class="info-item">
          <div class="info-label">K Level</div>
          <div class="info-value">${patient.k_level ?? 'N/A'} mEq/L</div>
        </div>
      </div>
    </div>

    <!-- Analysis Results -->
    <div class="section">
      <h2>Analysis Results</h2>
      <div class="info-grid">
        <div class="info-item">
          <div class="info-label">Analysis Date</div>
          <div class="info-value">${timestamp}</div>
        </div>
        <div class="info-item">
          <div class="info-label">Model Version</div>
          <div class="info-value">V3 Multilabel</div>
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Detected Condition</th>
            <th>Urgency Level</th>
          </tr>
        </thead>
        <tbody>
          ${conditionRows}
        </tbody>
      </table>

      ${analysis.notes ? `<p style="color: #666; font-size: 13px; margin-top: 10px;"><strong>Notes:</strong> ${analysis.notes}</p>` : ''}
    </div>

    <!-- Clinical Guidance -->
    <div class="section">
      <h2>Clinical Guidance</h2>
      <div style="background: #f9f9f9; padding: 15px; border-radius: 6px; font-size: 13px; line-height: 1.8;">
        <p><strong>Critical ⚠️:</strong> Requires immediate clinical evaluation and intervention.</p>
        <p><strong>High ⚠️:</strong> Prioritize clinical assessment; consider specialist referral.</p>
        <p><strong>Normal ✓:</strong> No acute findings; routine follow-up as appropriate.</p>
      </div>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">
      <strong>DISCLAIMER:</strong> This report is generated by an AI-assisted ECG analysis system for educational purposes only.
      It is NOT a medical diagnosis and is not FDA-cleared.
      Always consult a qualified healthcare professional for clinical decision-making.
      Patient privacy is protected in accordance with HIPAA regulations.
    </div>

    <!-- Footer -->
    <div class="footer">
      <p>EKG Intelligence Platform | Report generated on ${new Date().toISOString()}</p>
      <p>For clinical support, contact your healthcare provider.</p>
    </div>
  </div>
</body>
</html>
  `;
}
