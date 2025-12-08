from unittest.mock import patch
from ocr import run_ocr

def test_run_ocr_extract_text():
    fake_output = [
        [  
            [
                [[0, 0], [1, 1], [1, 1], [0, 1]],   # box
                ("Xin chào", 0.98)                  # (text, confidence)
            ]
        ]
    ]

    with patch('ocr.ocr_model.predict', return_value=fake_output):
        result = run_ocr("dummy.jpg")

    assert result == ["Xin chào"]