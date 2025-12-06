# Vize

![Flutter](https://img.shields.io/badge/Flutter-02569B?style=flat&logo=flutter&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)

> [!WARNING]
> **Work in Progress:** This project is currently under active development. Expect bugs and breaking changes.

**Vize** is an offline, privacy-first mobile system designed to extract, analyze, and translate real-world text between English and Vietnamese — strictly on-device.

This project is developed as a graduation thesis at **HCMUTE** (Ho Chi Minh City University of Technology and Education).

## Overview

Vize leverages lightweight AI models to bridge the language gap without relying on cloud APIs. By processing data locally, it ensures zero latency and absolute user privacy. The application focuses on three core pipelines:

1.  **Optical Character Recognition (OCR):** Detecting and reading Vietnamese/English text from images.
2.  **Semantic Analysis (NER):** Understanding context by extracting entities (Names, Dates, Locations).
3.  **Neural Machine Translation (NMT):** Translating text bi-directionally between English and Vietnamese.