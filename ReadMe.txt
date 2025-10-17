Hello!
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