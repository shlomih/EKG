import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_signal_from_image(image_path, debug=False):
    """Filters grid and extracts 1D signal from EKG scan."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")

    # 1. Convert to HSV to isolate the red/pink grid
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range for pink/red grid lines
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    grid_mask = cv2.inRange(hsv, lower_red, upper_red)

    # 2. Threshold for the black signal trace
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_trace = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Remove grid from trace
    cleaned_trace = cv2.bitwise_and(binary_trace, binary_trace, mask=cv2.bitwise_not(grid_mask))

    # 4. Extract 1D Signal (Column-wise median of black pixels)
    rows, cols = cleaned_trace.shape
    signal = []
    for x in range(cols):
        black_pixels = np.where(cleaned_trace[:, x] == 255)[0]
        if len(black_pixels) > 0:
            signal.append(np.median(black_pixels))
        else:
            signal.append(signal[-1] if signal else rows / 2)

    # Normalize: Invert Y (image 0 is top) and center around 0
    signal = np.array([rows - y for y in signal])
    signal = signal - np.mean(signal)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.plot(signal, color='#00E5B0')
        plt.title("Extracted Digital Signal")
        plt.show()

    return signal


if __name__ == "__main__":
    # Test with a placeholder or actual scan
    print("Digitization Pipeline Ready.")
