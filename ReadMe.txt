# ReLie Toolkit

Lightweight student toolset combining a **Lie Detection (CV)** demo and a **Port Scanner (learning mode)**.

## Features
- Real-time webcam demo with facial-landmark based predictions.  
- Per-frame confidence and CSV export for analysis.  
- Simple port scanner with configurable range and safe defaults.  
- Easy to run locally; minimal dependencies.

## Quick start
If Node.js not installed:
	winget install OpenJS.NodeJS
To Run the Project (Backend):
	cd path-to-the-folder-with-files
	pip install -r requirements.txt
	python app.py
In a separate CMD window (Frontend):
	cd path-to-the-folder-with-files
	npx serve Frontend

Before running the project, please ensure that ffmpeg is installed. If not:
	I have Given the file in Additional files with the name "ffmpeg-7.1.1-full_build"
	Copy the bin folder path (e.g., C:\ffmpeg-7.1.1-full_build\bin)
	Add it to your system PATH:
		Open Start → search "Environment Variables"
		Edit System Environment Variables → Click Environment Variables
		Under "System variables", find Path → Edit → Add new → Paste the bin path
	Test by running ffmpeg -version in a new terminal.

## Note / Ethics
This repo is for **educational use only**. Do **not** use the port scanner on systems you do not own or have permission to test. The lie-detection demo is a research prototype and not a reliable indicator of truth.

## Contact
GitHub: `https://github.com/L0stxRising` — PRs and issues welcome.


