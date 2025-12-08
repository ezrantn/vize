from paddleocr import PaddleOCR
import glob
import os

# Initialize the OCR model with specific tuning for Vietnamese
# 'vi' loads the Vietnamese model, which supports English + Vietnamese accents
# Disable document structure analysis to focus purely on text extraction
# Default is 960. We increase to 2500 to prevent the model 
# from shrinking high-res images, which causes small accents (dots, hooks) to blur.
# Bounding Box Expansion: Default is 1.5. We increase to 1.6 to slightly 
# expand the detected box. This ensures tall diacritics (like 'Â', 'ắ', 'ệ')
# are not cut off at the edges of the box.
ocr_model = PaddleOCR(
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=2000,
    text_det_unclip_ratio=1.6
)

def run_ocr(image_path: str):
    """
    Runs OCR on a single image and returns a list of text strings.
    Used primarily for Unit Testing or when you only need the raw text.
    """
    # predict() expects a list of images or a single path
    result = ocr_model.predict(img=image_path)
    texts = []

    # Handle cases where result might be None or empty
    if not result or result[0] is None:
        return texts

    # The first element of result is the list of detected lines
    for line in result[0]:
        texts.append(line[1][0])  # extract text only

    return texts


def process_batch(pattern: str = "./images/test_*_vn.jpg"):
    """
    Finds images matching the pattern, runs OCR, prints results to console,
    and saves annotated images/JSON to the output directory.
    """
    # Use glob to find all files matching the pattern
    image_paths = [os.path.normpath(p) for p in glob.glob(pattern)]
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        result = ocr_model.predict(img=img_path)

        # Create a specific folder name based on the image name
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join("output", base_name)
        
        # PaddleOCR returns a list of result objects
        for res in result:
            res.print()
            
            # Save results to disk
            # Ensure directory exists is handled internally by save_to_img usually, 
            # but usually good to ensure output root exists.
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            res.save_to_img(output_dir)
            res.save_to_json(output_dir)

    print("Batch processing complete.")