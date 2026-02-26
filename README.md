# Project Deployment – Image Processing Pipeline

This repository contains a simple core module plus an image dot‑reconstruction + digest pipeline.

## Files

- `core.py` – core status function
- `image_processing/main.py` – batch image dot reconstruction + digest
- `deploy.txt` – human instructions
- `image_processing/out/` – generated outputs (SVG/PNG + digest.json)

## Quick Start

1. Install Python 3 and `pip install pillow numpy`
2. Put your images into `image_processing/images/`
3. Run:
   - `python core.py`
   - `python image_processing/main.py`
4. See results in `image_processing/out/`
