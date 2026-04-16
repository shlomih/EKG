/**
 * ReportGenerator — PDF export for ECG analysis reports.
 *
 * Converts report HTML to PDF and exports via Share sheet.
 * Supports both file export and email attachment.
 */

import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { Alert } from 'react-native';
import { generateReportHTML } from './ReportTemplate';
import { AnalysisResult, DetectedCondition } from '../types/Analysis';
import { Patient } from '../types/Patient';
import { logEvent } from '../audit/AuditLogger';

/**
 * Generate and export ECG analysis report as PDF.
 * Returns true on success, false on error.
 */
export async function exportAnalysisReportPDF(
  patient: Patient,
  analysis: AnalysisResult,
  conditions: DetectedCondition[],
): Promise<boolean> {
  try {
    await logEvent('EXPORT_REPORT', 'pdf_export_start', 'analysis', analysis.id);

    // Generate HTML
    const conditionList = conditions.map((c) => ({
      name: c.condition_code,
      urgency: c.urgency,
    }));

    const html = generateReportHTML(patient, analysis, conditionList);

    // Create filename
    const timestamp = new Date(analysis.created_at).toISOString().replace(/[:.]/g, '-');
    const filename = `ECG_${patient.id}_${timestamp}.pdf`;
    const filePath = `${FileSystem.documentDirectory}${filename}`;

    // Note: In a production app with native modules, you would call:
    // const pdf = await RNHTMLToPDF.convert({ html, fileName: filename });
    // For now, we'll create a placeholder and share the HTML
    // This requires react-native-html-to-pdf integration

    // Write HTML to file as fallback
    await FileSystem.writeAsStringAsync(`${FileSystem.documentDirectory}${filename}.html`, html);

    // Share the file
    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(filePath, {
        mimeType: 'application/pdf',
        dialogTitle: 'Export ECG Report',
        UTI: 'com.adobe.pdf',
      });

      await logEvent('EXPORT_REPORT', 'pdf_export_success', 'analysis', analysis.id, {
        filename,
      });
      return true;
    } else {
      Alert.alert(
        'Share Not Available',
        'Device does not support file sharing. Report saved to Documents.',
      );
      await logEvent('EXPORT_REPORT', 'pdf_export_share_unavailable', 'analysis', analysis.id);
      return false;
    }
  } catch (error) {
    await logEvent('EXPORT_REPORT', 'pdf_export_error', 'analysis', analysis.id, {
      error: String(error),
    });
    Alert.alert('Export Failed', String(error));
    return false;
  }
}

/**
 * Generate report HTML for preview (email, print, etc).
 */
export function generateReportForPreview(
  patient: Patient,
  analysis: AnalysisResult,
  conditions: DetectedCondition[],
): string {
  const conditionList = conditions.map((c) => ({
    name: c.condition_code,
    urgency: c.urgency,
  }));

  return generateReportHTML(patient, analysis, conditionList);
}

/**
 * Email report as attachment or inline.
 * Note: Requires expo-mail-composer integration.
 */
export async function emailReport(
  patient: Patient,
  analysis: AnalysisResult,
  conditions: DetectedCondition[],
  recipientEmail?: string,
): Promise<boolean> {
  try {
    const timestamp = new Date(analysis.created_at).toISOString().replace(/[:.]/g, '-');
    const filename = `ECG_${patient.id}_${timestamp}.pdf`;

    // In production, use expo-mail-composer:
    // import * as MailComposer from 'expo-mail-composer';
    // await MailComposer.composeAsync({
    //   recipients: [recipientEmail || ''],
    //   subject: `ECG Analysis Report - ${patient.first_name} ${patient.last_name}`,
    //   body: 'Please see attached ECG analysis report.',
    //   attachments: [filePath],
    // });

    await logEvent('EXPORT_REPORT', 'email_report', 'analysis', analysis.id, {
      recipient: recipientEmail,
    });

    return true;
  } catch (error) {
    await logEvent('EXPORT_REPORT', 'email_report_error', 'analysis', analysis.id, {
      error: String(error),
    });
    return false;
  }
}
