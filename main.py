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
ocr = PaddleOCR(
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=3000,
    text_det_unclip_ratio=1.6
)

# Perform OCR on the image
# 1. Use glob to find all files matching the pattern
# This creates a list like ['./images/test_1_vn.jpg', './images/test_2_vn.jpg', ...]
image_paths = glob.glob("./images/test_*_vn.jpg")

print(f"Found {len(image_paths)} images to process.")

# 2. Loop through each image path
for img_path in image_paths:
    print(f"Processing: {img_path}")
    
    # Run prediction on the SINGLE current image
    result = ocr.predict(input=img_path)

    # 3. Handle results
    # We create a specific folder name based on the image name so results don't get mixed up
    # e.g., if file is "test_1_vn.jpg", output goes to "./output/test_1_vn/"
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join("output", base_name)
    
    for res in result:
        res.print()
        # Save to a specific sub-folder for this image
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

print("Batch processing complete.")