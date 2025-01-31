import os
import cv2
import pytesseract
import pandas as pd
import re

# Set Tesseract path for Mac (if needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'


# Preprocessing function
def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive thresholding (less aggressive)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )

    # Save preprocessed and grayscale images for debugging
    log_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_folder, exist_ok=True)
    grayscale_path = os.path.join(log_folder, f"grayscale_{os.path.basename(image_path)}")
    preprocessed_path = os.path.join(log_folder, f"preprocessed_{os.path.basename(image_path)}")
    cv2.imwrite(grayscale_path, gray)
    cv2.imwrite(preprocessed_path, adaptive_thresh)
    print(f"Saved grayscale image: {grayscale_path}")
    print(f"Saved preprocessed image: {preprocessed_path}")

    return adaptive_thresh


# OCR function
def extract_text(image):
    # Define Tesseract configuration
    config = '--psm 6 --oem 3'  # PSM 6: Assume a uniform block of text
    text = pytesseract.image_to_string(image, config=config, lang='spa')
    return text


# Parse text into structured data
def parse_text_to_table(text, category):
    rows = text.split("\n")
    structured_data = []

    if category == "Antecedentes":
        pattern = re.compile(
            r"(\d{7}-\d{2})\s+([A-Z]+)\s+([A-Z]+)\s+([A-Z .]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d,]+)\s+([A-Z]+)"
        )
        for row in rows:
            row = row.strip()
            match = pattern.match(row)
            if match:
                structured_data.append(match.groups())

    elif category == "Conglomerado":
        pattern = re.compile(
            r"(\d{7})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\w-]+)"
        )
        for row in rows:
            row = row.strip()
            match = pattern.match(row)
            if match:
                structured_data.append(match.groups())

    return structured_data


# Save structured data to CSV with predefined headers
def save_to_csv(data, output_path, category):
    if category == "Antecedentes":
        headers = [
            "Registration Number",
            "Last Name (Paternal)",
            "Last Name (Maternal)",
            "First Name(s)",
            "Folio Number",
            "Verbal Score",
            "Math Score",
            "Final Score",
            "Specific Tests",
            "Post Grades",
            "Admission Status"
        ]
    elif category == "Conglomerado":
        headers = [
            "Registration Number",
            "Score 1",
            "Score 2",
            "Score 3",
            "Score 4",
            "Remarks"
        ]

    # Ensure rows match expected columns
    formatted_data = []
    for row in data:
        while len(row) < len(headers):
            row.append("")
        formatted_data.append(row[:len(headers)])

    df = pd.DataFrame(formatted_data, columns=headers)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Data saved to {output_path}")


# Main workflow
def main():
    # Folder paths
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    antecedents_folder = os.path.join(data_folder, "Antecedentes")
    conglomerado_folder = os.path.join(data_folder, "Conglomerado")
    output_folder = os.path.join(data_folder, "output")
    log_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

    # Create output and logs folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Process each subfolder
    for folder, category in [(antecedents_folder, "Antecedentes"), (conglomerado_folder, "Conglomerado")]:
        for filename in os.listdir(folder):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(folder, filename)
                print(f"Processing {category} file: {filename}")

                # Preprocess the image
                processed_image = preprocess_image(image_path)

                # Extract text using OCR
                text = extract_text(processed_image)

                # Save raw OCR output for debugging
                raw_ocr_output_path = os.path.join(log_folder, f"ocr_output_{filename}.txt")
                with open(raw_ocr_output_path, "w", encoding="utf-8") as ocr_file:
                    ocr_file.write(text)
                print(f"Saved OCR output to: {raw_ocr_output_path}")

                # Parse text into structured data
                structured_data = parse_text_to_table(text, category)

                if structured_data:
                    # Save to CSV
                    output_filename = f"{os.path.splitext(filename)[0]}_{category}.csv"
                    output_path = os.path.join(output_folder, output_filename)
                    save_to_csv(structured_data, output_path, category)
                else:
                    print(f"No structured data extracted for {filename}. Check logs.")


# Run the script
if __name__ == "__main__":
    main()
