# A hobby object built with ollama, whisper and coqui-tts:
This was a early hobby object used to learn AI related libraries and explore around the different fields of components for an AI agent
This project uses Gradio as the GUI, supporting usage in Local Area Network, the computational power draws from the host device
- Whisper for STT and coqui for TTS, they provide a lightweighted solution, mainly chosen as cuda for 50 series wasn't available back then
- Ollama is used with schema as shown in the code, as the production workflow wasn't optimized at the point of time
- Lots of script used for modularization to allow quick replacement, allowing quick learning and test results
- This was also developed as a mother day's gift
- Memory system uses faiss index, but lacks memory decay mechanics, which might cause explosion and hallucination as logs accumulate
- memory summarize key facts, preferences, and decision but might expose redudancies accross sections.
---
At last, this is a small project showing the beginning of my learning in the AI architecture, where later I have learned more programming skills and tools, allowing better optimization
at the future, this might be used for training optimization skills, especially the memory sections before touching on big projects, this might include:
- Removing redudant and strengthening past memory
- Natural memory decay or archive on unrelevant / old recorded data
- Quick check and control on contradicting facts
- Confidence scoring for items without assurance
