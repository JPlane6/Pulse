# PULSE — Nurse Assistant Robot

An autonomous hospital triage robot that navigates corridors, enters patient rooms, holds a voice conversation to assess how the patient is doing, flags emergencies, and logs everything. Built on a Raspberry Pi with a MacBook acting as the AI brain over WiFi.

---

## What is PULSE?

PULSE is a nurse assistant robot designed to handle the first step of patient triage in a hospital setting. Instead of a nurse walking room to room to do initial check-ins, PULSE does it autonomously. It drives itself down the corridor, detects when it's passing a patient room, enters, greets the patient by name, asks up to 6 clinically relevant questions using AI-generated follow-ups, decides if the situation is urgent, alerts a nurse if it is, and saves a full record of the session before continuing to the next room.

The robot handles all the physical work on a Raspberry Pi — driving, listening, speaking, reading distances with a LiDAR sensor, and showing status on an LCD screen. The heavy AI work — converting speech to text and generating intelligent follow-up questions — is offloaded to a MacBook server running on the same WiFi network.

---

## How It Works

1. The robot boots up, runs a self-check, and confirms the microphone, speaker, and AI server are all reachable before doing anything. If something is missing it says so out loud and stops safely.

2. It moves forward down the corridor using a LiDAR sensor to track the distance to the left and right walls as it drives.

3. When one of those wall distances suddenly increases, the robot knows the wall ended and a room opened up. It turns toward the opening and drives in.

4. Once inside, it greets the patient by name and starts the assessment. The first question is always a pain scale from 1 to 10.

5. After each answer, the recording is sent to the MacBook, which converts it to text using Whisper and sends the text back. The AI then reads the full conversation so far and decides the single most important follow-up question to ask next.

6. This continues for up to 6 questions total. The AI can also decide it has enough information early and stop on its own.

7. At the end, the AI makes a final yes or no decision on whether the patient should be flagged as urgent. If yes, the robot tells the patient a nurse is on the way. If no, it tells the patient their status has been recorded.

8. The full session — patient name, room, date, every question and answer, and the final triage result — is saved as a JSON log file.

9. The robot exits the room and resumes navigating the corridor to find the next room.

---

## System Layout

The robot and the AI server are two separate machines talking to each other over WiFi.

The Raspberry Pi handles everything physical. It controls the motors through an Arduino, reads the LiDAR sensor for navigation, records audio from the USB headset, plays back speech using Piper TTS, and shows live status on the LCD screen. It also saves all the session logs.

The MacBook handles all the AI. It runs a Flask web server that the Pi sends requests to. When the Pi records an answer, it sends the audio file to the MacBook, which runs it through Whisper to get the text, then passes the conversation history to Phi3:mini via Ollama to get the next question. The MacBook sends the results back and the Pi continues the session.

---

## Hardware

| Part | What It Does |
|---|---|
| Raspberry Pi 5 | Runs the robot — navigation, audio, motors, display, logging |
| MacBook M2 (16GB RAM) | Runs the AI server — speech to text and question generation |
| YDLidar X4 | Reads distances in all directions for navigation and room detection |
| Arduino Mega + Adafruit Motor Shield | Controls the four drive motors based on commands from the Pi |
| Logitech H570e USB Headset | Microphone for recording patient answers, speaker for TTS playback |
| 20x4 I2C LCD Screen | Shows live distances, navigation status, and triage results |

---

## File Structure

The files that matter most and what they actually do:

main.py is the full autonomous program. It runs navigation, detects room openings, enters rooms, and kicks off the triage session.

client.py is the triage session itself. It handles greeting the patient, recording answers, talking to the server, and saving the log. This can also be run on its own without navigation for testing.

server.py runs on the MacBook. It receives audio from the Pi, runs Whisper to transcribe it, calls Phi3:mini to generate questions, and sends everything back.

motorControl_Pi.py sends drive commands to the Arduino over a serial connection.

lidarHelpers.py handles all the LiDAR logic. It divides the sensor readings into three zones — front, left, and right — and returns clean distances for the navigation code to use.

modules/config.json is where you set the patient name and room number before each run.

modules/logs/ is where all session records are saved, one folder per session plus a combined file with every session ever recorded.

The motorControl_Arduino folder contains the firmware that runs on the Arduino and controls the actual motors.

---

## Setting Up the MacBook

Start here before touching the Pi. The Pi won't work without the server running.

First, make sure you have Homebrew installed. Then install ffmpeg, which Whisper needs to decode audio files. Without it Whisper will fail silently on certain audio formats.

Next, install Ollama and use it to download Phi3:mini. Phi3:mini is the AI model that generates the triage questions and makes the final urgency decision. It's about 2.3GB. Once it's downloaded, start the Ollama service so it's running in the background before you launch the server.

Then install the Python packages from requirements_laptop.txt and start server.py. When it's running you'll see it confirm that Whisper loaded and that it's listening on port 5001.

Note on LLaVA: the codebase has some references to LLaVA, a vision model that was originally used to visually assess the patient via camera. It was cut from the active pipeline because running LLaVA (4.9GB) and Phi3:mini (2.3GB) at the same time on a 16GB MacBook caused the system to hit swap and become unusably slow. Do not pull or run LLaVA unless you have significantly more RAM.

Finally, find your MacBook's local IP address on the WiFi network. You'll need to paste it into client.py on the Pi so the two machines can talk.

---

## Setting Up the Raspberry Pi

Install the system packages first before anything else. You need i2c-tools for the LCD screen, and python3-dev plus cmake plus swig to be able to build the YDLidar SDK from source. These cannot be skipped.

Enable I2C in raspi-config so the Pi can talk to the LCD over the I2C bus. After enabling it, run i2cdetect to confirm the LCD is wired correctly. You should see the address 27 show up in the grid. If it's blank, check your wiring.

Install Piper TTS and download the voice model files from Hugging Face. The model path in client.py needs to match wherever you put the files on your Pi.

Build and install the YDLidar SDK from source using the GitHub repo. This is the driver that lets Python talk to the LiDAR sensor.

Install the Python packages from requirements_pi.txt.

Flash the Arduino firmware from the motorControl_Arduino folder using the Arduino IDE.

Set up udev rules to give the LiDAR and Arduino stable device names that don't change on reboot. Without this, the LiDAR might be on a different port every time you restart and the code won't find it. The code expects the LiDAR at /dev/lidar and the Arduino at /dev/arduino.

Update the SERVER_URL line in client.py with your MacBook's IP address.

Edit modules/config.json to set the room number and patient name for the upcoming session.

Plug in the USB headset and verify it shows up in both the playback list and the recording list using the aplay and arecord commands. The Pi uses two separate audio systems — one for playback and one for recording — and the headset needs to appear in both.

---

## Known Limitations

The visual triage feature is disabled. The robot originally used a camera and the LLaVA vision model to visually assess the patient's condition before asking questions. This was cut because running LLaVA alongside Phi3:mini and Whisper on a 16GB MacBook caused memory issues. Re-enabling it would require either more RAM or running the models on separate machines.

Room detection can miss fast openings. The robot detects rooms by watching for a sudden jump in side wall distance. If the robot is moving too quickly, it can drive past a doorway before the LiDAR picks up the change. If rooms are being missed during testing, slow the robot down.

Phi3:mini sometimes adds extra text after the question. The server takes only the first line of every response as a safeguard. This works well in practice but if you change the prompt significantly, test it to make sure the first line is always the clean question and nothing else.

The audio pipeline uses two different systems. The TTS playback uses ALSA (the Linux audio layer) and the microphone recording uses PortAudio (a cross-platform audio library). These are completely separate and the headset needs to be detected by both independently. client.py handles this automatically and prints a full device scan on startup so you can see exactly what it found.

The Arduino locks after an emergency stop. When the robot sends a STOP command, the Arduino enters a locked state and won't accept any movement commands until it receives a GO command. This is intentional for safety but can be confusing during manual testing.

Whisper on Apple Silicon requires fp16 to be disabled. The M2 GPU backend in PyTorch does not support fp16 for Whisper. The server has this set correctly already — don't change it.

---

## Future Work

Re-enabling visual triage via camera once a higher RAM machine or a separate inference server is available.

A web dashboard for viewing and searching patient session logs instead of reading raw JSON files.

Multi-robot support so several PULSE units can share the same server and log to the same database without conflicts.

Automatic room assignment by reading a room schedule or hospital system instead of manually editing config.json before each session.

Wake word detection so the robot waits for the patient to signal they're ready before starting the assessment, rather than beginning immediately on entry.
