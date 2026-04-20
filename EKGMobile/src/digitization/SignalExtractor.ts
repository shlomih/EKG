/**
 * SignalExtractor — Option B: Guided Capture + Scale-Based Extraction
 *
 * Takes a camera photo aligned within the guide frame overlay (fixed geometry)
 * and extracts 12-lead ECG signals from fixed lead positions.
 *
 * HIPAA 5.3.6: All image processing done in memory; temp image deleted immediately.
 *
 * What Copilot handles:
 * - Load image from file
 * - Crop into fixed lead rectangles (known paper geometry)
 * - Basic pixel-intensity-to-voltage conversion with known scale
 *
 * What Claude handles (TODO):
 * - Signal tracing (darkest pixel per column = signal sample)
 * - Baseline correction (isoelectric line from TP segment)
 * - Calibration pulse detection or fixed 10mm/mV assumption
 * - Resample to exactly 5000 samples @ 500 Hz
 */

import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import { secureDelete } from '@/src/utils/TempFileManager';

export interface SignalExtractionResult {
  signal: Float32Array; // Shape (12, 5000) flattened
  sampleRate: number;
  calibrationScale: number; // mm to mV conversion
  extractedAt: string;
}

/**
 * Extract 12-lead ECG signal from aligned camera photo.
 *
 * @param imageUri - Temp file path to JPEG photo
 * @param paperSize - 'a4' | 'letter' | 'thermal' (determines overlay geometry)
 * @returns Float32Array[60000] = 12 leads × 5000 samples
 */
export async function extractSignal(
  imageUri: string,
  paperSize: 'a4' | 'letter' | 'thermal',
): Promise<SignalExtractionResult> {
  try {
    // Step 1: Load and get image metadata
    const imageAsset = await FileSystem.getInfoAsync(imageUri);
    if (!imageAsset.exists) {
      throw new Error('Image file not found');
    }

    // Step 2: Load image and get dimensions
    // Note: In real usage, you'd use ImageManipulator to resize if needed
    // For now, assume image is camera output from CameraCapture
    const base64 = await FileSystem.readAsStringAsync(imageUri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    if (!base64) {
      throw new Error('Failed to read image');
    }

    // Step 3: Get image dimensions
    // TODO: Use expo-image-manipulator or Image.getSize() to get true dimensions
    // For MVP, assume standard camera photo: ~1080×1920 or similar
    const imageDimensions = await getImageDimensions(imageUri);

    // Step 4: Calculate paper overlay geometry on image
    const paperGeometry = calculatePaperGeometry(
      paperSize,
      imageDimensions.width,
      imageDimensions.height,
    );

    // Step 5: Crop and extract each lead region
    const signal = new Float32Array(12 * 5000);
    const leadPositions = getLead12Positions(paperSize);

    for (let leadIdx = 0; leadIdx < 12; leadIdx++) {
      const leadPos = leadPositions[leadIdx];
      if (!leadPos) continue;

      // Translate from paper mm to image pixels
      const pixelCrop = mmToPixels(leadPos, paperGeometry);

      try {
        // Extract this lead's image region
        const leadImageBase64 = await cropImageRegion(
          base64,
          pixelCrop,
          imageDimensions,
        );

        // TODO (Claude): Signal tracing logic
        // const leadSignal = traceLeadSignal(leadImageBase64, pixelCrop);
        // For MVP, generate placeholder (will be replaced by Claude)
        const leadSignal = generatePlaceholderLead();

        // Copy to output array
        signal.set(leadSignal, leadIdx * 5000);
      } catch (error) {
        console.warn(`Failed to extract lead ${leadIdx}:`, error);
        // Use placeholder signal for this lead
        signal.set(generatePlaceholderLead(), leadIdx * 5000);
      }
    }

    return {
      signal,
      sampleRate: 500,
      calibrationScale: 10, // mm/mV (standard ECG paper)
      extractedAt: new Date().toISOString(),
    };
  } finally {
    // Always securely delete temp image
    await secureDelete(imageUri);
  }
}

// ============================================================================
// HELPER FUNCTIONS — Copilot
// ============================================================================

/**
 * Get image dimensions from file (requires native bridge or canvas).
 * TODO: Use react-native Image.getSize() or expo-image for real implementation.
 */
async function getImageDimensions(
  imageUri: string,
): Promise<{ width: number; height: number }> {
  // MVP: Return typical camera photo dimensions
  // In production, use Image.getSize(imageUri, (w, h) => ...)
  return { width: 1080, height: 1920 };
}

/**
 * Standard 12-lead paper layout:
 * Row 1 (0-15mm):    I,    II,   III   (top 3 leads)
 * Row 2 (20-35mm):   aVR,  aVL,  aVF   (augmented limb leads)
 * Row 3 (40-55mm):   V1,   V2,   V3    (precordial leads left)
 * Row 4 (60-75mm):   V4,   V5,   V6    (precordial leads right)
 *
 * Within each row: columns at 0mm, 70mm, 140mm (3 leads × 70mm width each)
 *
 * Paper width: 210mm (A4) or ~216mm (Letter)
 */
interface LeadPosition {
  leadIndex: number; // 0-11
  name: string; // 'I', 'II', ..., 'V6'
  row: number; // 0-3 (which 15mm band)
  col: number; // 0-2 (which 70mm column)
  mmX: number; // mm offset from left
  mmY: number; // mm offset from top
}

function getLead12Positions(paperSize: 'a4' | 'letter' | 'thermal'): LeadPosition[] {
  // Standard ECG 12-lead layout (3×4 or 2×6)
  // Using 3-column layout with 4 rows
  const leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

  const positions: LeadPosition[] = [];
  const colWidth = 70; // mm per column
  const rowHeight = 15; // mm per row

  for (let i = 0; i < 12; i++) {
    const col = i % 3;
    const row = Math.floor(i / 3);

    positions.push({
      leadIndex: i,
      name: leadNames[i],
      row,
      col,
      mmX: col * colWidth + 10, // 10mm left margin
      mmY: row * rowHeight + 20, // 20mm top margin
    });
  }

  return positions;
}

/**
 * Calculate paper overlay geometry on image.
 * Maps: paper mm → image pixels
 */
interface PaperGeometry {
  paperWidthMm: number;
  paperHeightMm: number;
  imageLeft: number; // pixel offset of paper left edge
  imageTop: number; // pixel offset of paper top edge
  scaleMmToPixels: number; // pixels per mm
}

function calculatePaperGeometry(
  paperSize: 'a4' | 'letter' | 'thermal',
  imageWidth: number,
  imageHeight: number,
): PaperGeometry {
  // Paper dimensions
  const paperDims: Record<string, { width: number; height: number }> = {
    a4: { width: 210, height: 297 },
    letter: { width: 215.9, height: 279.4 },
    thermal: { width: 127, height: 200 },
  };

  const dims = paperDims[paperSize];

  // Assume image is centered in camera view with 10% margins on each side
  const effectiveWidth = imageWidth * 0.8;
  const effectiveHeight = imageHeight * 0.8;
  const imageLeft = imageWidth * 0.1;
  const imageTop = imageHeight * 0.1;

  // Calculate scale factor (pixels per mm)
  const scaleX = effectiveWidth / dims.width;
  const scaleY = effectiveHeight / dims.height;
  const scale = Math.min(scaleX, scaleY); // Use smaller to fit both dimensions

  return {
    paperWidthMm: dims.width,
    paperHeightMm: dims.height,
    imageLeft,
    imageTop,
    scaleMmToPixels: scale,
  };
}

/**
 * Convert paper mm coordinates to image pixel coordinates.
 */
interface PixelCrop {
  x: number; // pixel left
  y: number; // pixel top
  width: number; // pixel width
  height: number; // pixel height
}

function mmToPixels(leadPos: LeadPosition, geom: PaperGeometry): PixelCrop {
  // Lead region is 70mm wide × 40mm tall (for clarity)
  // Actual trace area may be smaller (e.g., 65mm × 35mm)
  const leadWidthMm = 65; // actual signal area (mm)
  const leadHeightMm = 35; // actual signal area (mm)

  return {
    x: Math.round(geom.imageLeft + leadPos.mmX * geom.scaleMmToPixels),
    y: Math.round(geom.imageTop + leadPos.mmY * geom.scaleMmToPixels),
    width: Math.round(leadWidthMm * geom.scaleMmToPixels),
    height: Math.round(leadHeightMm * geom.scaleMmToPixels),
  };
}

/**
 * Crop image region and return as base64.
 * TODO (Copilot): Use expo-image-manipulator to crop.
 */
async function cropImageRegion(
  _imageBase64: string,
  _crop: PixelCrop,
  _imageDims: { width: number; height: number },
): Promise<string> {
  // MVP: Return placeholder
  // In production:
  // const manipulated = await ImageManipulator.manipulateAsync(
  //   imageUri,
  //   [{ crop: { originX: crop.x, originY: crop.y, width: crop.width, height: crop.height } }],
  //   { compress: 0.9, format: ImageManipulator.SaveFormat.JPEG }
  // );
  // return manipulated.base64;

  return _imageBase64; // Placeholder
}

// ============================================================================
// PLACEHOLDER — To be replaced by Claude
// ============================================================================

/**
 * Generate placeholder lead signal (normal sinus rhythm pattern).
 * TODO (Claude): Replace with actual signal tracing from image pixel data.
 */
function generatePlaceholderLead(): Float32Array {
  const SAMPLES = 5000;
  const signal = new Float32Array(SAMPLES);
  const heartRate = 60; // bpm
  const samplingRate = 500; // Hz
  const cycleLength = (60 / heartRate) * samplingRate; // samples per cycle

  for (let i = 0; i < SAMPLES; i++) {
    const t = (i % cycleLength) / cycleLength;

    // Simple sinusoidal ECG-like pattern
    let value = 0;

    // P-wave
    if (t < 0.2) {
      value += 0.05 * Math.sin(Math.PI * t / 0.2);
    }
    // QRS complex
    if (t >= 0.2 && t < 0.3) {
      value += 1.0 * Math.sin(Math.PI * (t - 0.2) / 0.1);
    }
    // T-wave
    if (t >= 0.3 && t < 0.5) {
      value += 0.15 * Math.sin(Math.PI * (t - 0.3) / 0.2);
    }

    signal[i] = value;
  }

  return signal;
}
