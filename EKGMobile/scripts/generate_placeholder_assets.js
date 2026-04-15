/**
 * Generate placeholder asset PNGs for development.
 * Run: node scripts/generate_placeholder_assets.js
 *
 * Creates minimal valid PNG files so Expo doesn't error on startup.
 * Replace with real app icons before release.
 */
const fs = require('fs');
const path = require('path');

// Minimal 1x1 pixel green (#00E5B0) PNG
// This is a valid PNG file — the smallest possible
const PNG_1x1 = Buffer.from([
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
  0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
  0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
  0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
  0x54, 0x08, 0xD7, 0x63, 0x60, 0x60, 0x60, 0x00, // compressed data
  0x00, 0x00, 0x04, 0x00, 0x01, 0x27, 0x34, 0x27, //
  0x0A, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
  0x44, 0xAE, 0x42, 0x60, 0x82,
]);

const assetsDir = path.join(__dirname, '..', 'assets');
fs.mkdirSync(assetsDir, { recursive: true });

for (const name of ['icon.png', 'adaptive-icon.png', 'splash-icon.png']) {
  const dest = path.join(assetsDir, name);
  if (!fs.existsSync(dest)) {
    fs.writeFileSync(dest, PNG_1x1);
    console.log(`Created ${name}`);
  } else {
    console.log(`${name} already exists — skipping`);
  }
}
console.log('Done. Replace these with real icons before release.');
