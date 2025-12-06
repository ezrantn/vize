from paddleocr import PaddleOCR

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
result = ocr.predict(input="./images/test_1_vn.jpg")

# Process and Save Results
# The 'result' object contains methods to visualize and serialize the data directly.
# Print detected text and confidence scores to console
for res in result:
    res.print()
    
    # Save the image with bounding boxes drawn (creates ./output/ folder)
    res.save_to_img("output")
    
    # Save the raw data (coordinates, text, confidence) to a JSON file
    res.save_to_json("output")

print("Processing complete. Check the 'output' directory.")