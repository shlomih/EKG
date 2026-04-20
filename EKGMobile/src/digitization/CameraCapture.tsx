/**
 * CameraCapture — photo capture with alignment guides and paper-size selector.
 * HIPAA 5.3.6: All captured images processed in memory only, no gallery access.
 *
 * Features:
 * - White rectangle overlay matching ECG paper dimensions
 * - Paper-size selector: A4 (210×297mm) / US Letter (8.5×11in) / Thermal (5in width)
 * - Corner alignment arrows
 * - Capture button with loading state
 * - Returns { imageUri, paperSize } to parent
 */

import { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  Alert,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import { createTempPath, secureDelete } from '@/src/utils/TempFileManager';

export interface CameraResult {
  imageUri: string;
  paperSize: 'a4' | 'letter' | 'thermal';
  aspectRatio: number;
}

interface CameraCaptureProps {
  onCapture: (result: CameraResult) => void;
  onCancel: () => void;
}

// Paper dimensions (width × height in mm)
const PAPER_SIZES = {
  a4: { name: 'A4', width: 210, height: 297, ratio: 210 / 297 },
  letter: { name: 'US Letter', width: 215.9, height: 279.4, ratio: 215.9 / 279.4 },
  thermal: { name: 'Thermal (5")', width: 127, height: 200, ratio: 127 / 200 }, // ~5" width
};

export default function CameraCapture({ onCapture, onCancel }: CameraCaptureProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const [paperSize, setPaperSize] = useState<'a4' | 'letter' | 'thermal'>('a4');
  const [isCapturing, setIsCapturing] = useState(false);
  const [showPaperSelector, setShowPaperSelector] = useState(false);

  const screenWidth = Dimensions.get('window').width;
  const screenHeight = Dimensions.get('window').height;

  // Request camera permission
  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera access required to capture ECG strips.</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Camera Access</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
            <Text style={styles.cancelButtonText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const handleCapture = async () => {
    if (!cameraRef.current || isCapturing) return;

    setIsCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.9,
        base64: false, // Save to file instead of base64
      });

      if (!photo) {
        Alert.alert('Capture Failed', 'Could not capture photo');
        setIsCapturing(false);
        return;
      }

      // Move to secure temp file location
      const tempPath = createTempPath('jpg');
      await FileSystem.copyAsync({
        from: photo.uri,
        to: tempPath,
      });

      // Clean up original temp file from camera
      try {
        await FileSystem.deleteAsync(photo.uri, { idempotent: true });
      } catch {
        // Best effort cleanup
      }

      // Return result
      onCapture({
        imageUri: tempPath,
        paperSize,
        aspectRatio: PAPER_SIZES[paperSize].ratio,
      });
    } catch (error) {
      Alert.alert('Capture Error', String(error));
      setIsCapturing(false);
    }
  };

  // Calculate overlay dimensions to match paper aspect ratio
  const paperDims = PAPER_SIZES[paperSize];
  const maxWidth = screenWidth * 0.9;
  const maxHeight = screenHeight * 0.7;

  let overlayWidth = maxWidth;
  let overlayHeight = overlayWidth / paperDims.ratio;

  if (overlayHeight > maxHeight) {
    overlayHeight = maxHeight;
    overlayWidth = overlayHeight * paperDims.ratio;
  }

  const overlayLeft = (screenWidth - overlayWidth) / 2;
  const overlayTop = (screenHeight * 0.5 - overlayHeight) / 2 + 40; // Offset for controls below

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <CameraView style={styles.camera} ref={cameraRef} facing="back">
        {/* Alignment Overlay */}
        <View style={styles.overlayContainer}>
          {/* Guide Frame */}
          <View
            style={[
              styles.guideFrame,
              {
                width: overlayWidth,
                height: overlayHeight,
                left: overlayLeft,
                top: overlayTop,
              },
            ]}
          >
            {/* Corner Markers */}
            <CornerMarker position="top-left" />
            <CornerMarker position="top-right" />
            <CornerMarker position="bottom-left" />
            <CornerMarker position="bottom-right" />

            {/* Center Alignment Text */}
            <View style={styles.alignmentText}>
              <Text style={styles.alignmentTextContent}>Keep paper flat and centered</Text>
            </View>
          </View>

          {/* Dark overlay outside guide frame (darken edges) */}
          <View style={[styles.darkOverlay, { top: 0, height: overlayTop }]} />
          <View
            style={[
              styles.darkOverlay,
              { top: overlayTop + overlayHeight, height: screenHeight - (overlayTop + overlayHeight) },
            ]}
          />
          <View
            style={[
              styles.darkOverlay,
              { top: overlayTop, left: 0, width: overlayLeft, height: overlayHeight },
            ]}
          />
          <View
            style={[
              styles.darkOverlay,
              {
                top: overlayTop,
                left: overlayLeft + overlayWidth,
                width: screenWidth - (overlayLeft + overlayWidth),
                height: overlayHeight,
              },
            ]}
          />
        </View>
      </CameraView>

      {/* Bottom Control Panel */}
      <View style={styles.controlPanel}>
        {/* Paper Size Selector */}
        {showPaperSelector ? (
          <View style={styles.paperSelectorOpen}>
            {(Object.entries(PAPER_SIZES) as Array<[keyof typeof PAPER_SIZES, typeof PAPER_SIZES.a4]>).map(
              ([key, dims]) => (
                <TouchableOpacity
                  key={key}
                  style={[
                    styles.paperOption,
                    paperSize === key && styles.paperOptionActive,
                  ]}
                  onPress={() => {
                    setPaperSize(key);
                    setShowPaperSelector(false);
                  }}
                >
                  <Text
                    style={[
                      styles.paperOptionText,
                      paperSize === key && styles.paperOptionTextActive,
                    ]}
                  >
                    {dims.name}
                  </Text>
                </TouchableOpacity>
              ),
            )}
          </View>
        ) : (
          <TouchableOpacity
            style={styles.paperSelectorButton}
            onPress={() => setShowPaperSelector(true)}
          >
            <Text style={styles.paperSelectorButtonText}>
              📄 {PAPER_SIZES[paperSize].name}
            </Text>
          </TouchableOpacity>
        )}

        {/* Action Buttons */}
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.cancelButton}
            onPress={onCancel}
            disabled={isCapturing}
          >
            <Text style={styles.cancelButtonText}>Cancel</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.captureButton, isCapturing && styles.captureButtonDisabled]}
            onPress={handleCapture}
            disabled={isCapturing}
          >
            {isCapturing ? (
              <ActivityIndicator color="#071312" />
            ) : (
              <Text style={styles.captureButtonText}>📷 Capture</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

interface CornerMarkerProps {
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

function CornerMarker({ position }: CornerMarkerProps) {
  const positionStyles: Record<string, object> = {
    'top-left': { top: -8, left: -8 },
    'top-right': { top: -8, right: -8 },
    'bottom-left': { bottom: -8, left: -8 },
    'bottom-right': { bottom: -8, right: -8 },
  };

  const rotationStyles: Record<string, string> = {
    'top-left': '0deg',
    'top-right': '90deg',
    'bottom-right': '180deg',
    'bottom-left': '270deg',
  };

  return (
    <View
      style={[
        styles.cornerMarker,
        positionStyles[position],
        { transform: [{ rotate: rotationStyles[position] }] },
      ]}
    >
      <Text style={styles.cornerMarkerText}>⌜</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071312',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
  },
  guideFrame: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#FFFFFF',
    backgroundColor: 'transparent',
  },
  darkOverlay: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  alignmentText: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  alignmentTextContent: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '500',
    textAlign: 'center',
  },
  cornerMarker: {
    position: 'absolute',
    width: 16,
    height: 16,
  },
  cornerMarkerText: {
    fontSize: 24,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  controlPanel: {
    backgroundColor: '#0D1F1E',
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 20,
    borderTopWidth: 1,
    borderTopColor: '#1E3533',
  },
  paperSelectorButton: {
    backgroundColor: '#1E3533',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 12,
    alignItems: 'center',
  },
  paperSelectorButtonText: {
    color: '#00E5B0',
    fontSize: 14,
    fontWeight: '600',
  },
  paperSelectorOpen: {
    flexDirection: 'row',
    marginBottom: 12,
    gap: 8,
  },
  paperOption: {
    flex: 1,
    backgroundColor: '#1E3533',
    paddingHorizontal: 8,
    paddingVertical: 10,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1E3533',
  },
  paperOptionActive: {
    backgroundColor: '#00E5B0',
    borderColor: '#00E5B0',
  },
  paperOptionText: {
    color: '#5A8A85',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  paperOptionTextActive: {
    color: '#071312',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  captureButton: {
    flex: 1,
    backgroundColor: '#00E5B0',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  captureButtonDisabled: {
    opacity: 0.6,
  },
  captureButtonText: {
    color: '#071312',
    fontSize: 16,
    fontWeight: '700',
  },
  cancelButton: {
    flex: 0.3,
    backgroundColor: '#FF4444',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cancelButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  permissionText: {
    color: '#FFFFFF',
    fontSize: 16,
    marginBottom: 24,
    textAlign: 'center',
  },
  permissionButton: {
    backgroundColor: '#00E5B0',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  permissionButtonText: {
    color: '#071312',
    fontSize: 14,
    fontWeight: '600',
  },
});
