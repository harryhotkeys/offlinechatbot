Physics Chatbot

A simple AI-powered Physics Chatbot built with Python, Tkinter, and Sentence Transformers. The chatbot uses semantic similarity instead of traditional keyword matching, allowing it to understand different ways users might ask physics-related questions.

Features
Semantic question matching using Sentence Transformers
User-friendly graphical interface with Tkinter
Covers multiple physics topics:
Forces and Motion
Energy
Electricity
Heat and Temperature
Waves and Sound
Light and Optics
Matter and Atoms
Displays match confidence scores
Supports both English and some Greek keywords
Clear Chat functionality
Technologies Used
Python 3
Tkinter (GUI)
Sentence Transformers
PyTorch
all-MiniLM-L6-v2 embedding model
Installation
1. Clone the repository
git clone https://github.com/yourusername/physics-chatbot.git
cd physics-chatbot
2. Install dependencies
pip install sentence-transformers torch
Running the Application

Run the chatbot with:

python ChatbotReady.py

The GUI window will open automatically.

How It Works
Semantic Matching

Instead of checking for exact keywords, the chatbot:

Converts all predefined questions into embeddings.
Converts the user's input into an embedding.
Computes cosine similarity between them.
Returns the answer with the highest similarity score.
similarities = util.cos_sim(input_embedding, question_embeddings)[0]
best_idx = similarities.argmax().item()
Confidence Threshold

The chatbot uses a threshold value:

THRESHOLD = 0.3

If the similarity score is below this threshold, the chatbot responds with:

Sorry, I don't understand. Try asking about sound, atoms, or atom matter!
Supported Topics
Forces and Motion
Force
Motion
Speed
Distance
Time
Acceleration
Friction
Gravity
Mass
Inertia
Newton's Laws
Momentum
Free Fall
Energy
Work
Power
Kinetic Energy
Potential Energy
Mechanical Energy
Renewable Energy
Non-Renewable Energy
Conservation of Energy
Electricity
Current
Voltage
Resistance
Ohm's Law
Series Circuits
Parallel Circuits
Batteries
Conductors
Insulators
Heat and Temperature
Heat Transfer
Conduction
Convection
Radiation
Temperature Measurement
Waves and Sound
Sound Waves
Frequency
Pitch
Echoes
Wave Properties
Light and Optics
Reflection
Refraction
Mirrors
Lenses
Transparency
Matter and Atoms
Atoms
Molecules
Density
States of Matter
Evaporation
Condensation
Melting and Freezing
User Interface

The application includes:

Scrollable chat window
Text input field
Send button
Enter key support
Match confidence display
Clear Chat button

Example:

You: What is gravity?
Match confidence: 0.87
Bot: Gravity pulls objects toward Earth.
Project Structure
physics-chatbot/
│
├── ChatbotReady.py
├── README.md
└── requirements.txt
Example Questions
What is force?
Explain gravity.
What is kinetic energy?
How does a battery work?
What is refraction?
Tell me about atoms.
What is density?
Future Improvements
Add more physics concepts and explanations
Support follow-up conversations
Integrate formula solving
Add voice input/output
Support multiple languages
Store chat history
Connect to larger language models for richer explanations
License

This project is open source and available under the MIT License.

Author

Physics Chatbot created using Python, Tkinter, and Sentence Transformers for educational purposes.
