from logging import root

from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import scrolledtext

model = SentenceTransformer('all-MiniLM-L6-v2')


qa_pairs = [
    ("hello hi hey greetings καλησπερουδια γεια χαρά", "Hi! How can I help with physics today?"),

    # Forces and motion
    ("force δύναμη push pull newton", "Force is a push or pull on an object and is measured in Newtons (N)."),
    ("motion movement moving travel", "Motion happens when an object changes position over time."),
    ("speed velocity fast slow", "Speed = distance / time."),
    ("distance length travel path", "Distance is the total length an object travels."),
    ("time seconds minutes hours", "Time tells us how long an event lasts."),
    ("acceleration increase decrease change", "Acceleration is how quickly velocity changes."),
    ("friction rough surface slide", "Friction is a force that opposes motion between surfaces."),
    ("gravity earth fall weight", "Gravity pulls objects toward Earth."),
    ("mass weight heavy light", "Mass is the amount of matter in an object. Weight depends on gravity."),
    ("inertia newton first law", "Inertia is an object's tendency to resist changes in motion."),
    ("newton laws motion", "Newton's laws explain how forces affect motion."),
    ("balanced unbalanced forces", "Balanced forces cancel out. Unbalanced forces change motion."),
    ("momentum collision moving", "Momentum depends on mass and velocity."),
    ("freefall falling object", "In free fall, gravity is the only force acting on an object."),

    # Energy
    ("energy work power joules", "Energy is the ability to do work and is measured in Joules (J)."),
    ("work force distance", "Work happens when a force moves an object."),
    ("power watts energy", "Power measures how quickly work is done."),
    ("kinetic movement moving energy", "Kinetic energy is energy of motion."),
    ("potential stored height energy", "Potential energy is stored energy due to position."),
    ("mechanical energy", "Mechanical energy is kinetic plus potential energy."),
    ("renewable solar wind hydro", "Renewable energy comes from naturally replenished sources."),
    ("nonrenewable fossil fuel coal oil", "Non-renewable energy sources can run out."),
    ("conservation energy", "Energy cannot be created or destroyed, only transformed."),

    # Electricity
    ("electricity current voltage circuit", "Electric current is the flow of electric charges."),
    ("current electric charge flow", "Current is measured in Amperes (A)."),
    ("voltage electric potential", "Voltage is the difference in electrical potential."),
    ("resistance resistor ohm", "Resistance opposes current and is measured in Ohms (Ω)."),
    ("ohm voltage current formula", "Ohm's law: V = I × R."),
    ("series circuit battery", "In a series circuit, components are connected in one path."),
    ("parallel circuit", "In a parallel circuit, current can follow multiple paths."),
    ("battery electric source", "A battery provides electrical energy."),
    ("conductor copper metal", "Conductors allow electricity to flow easily."),
    ("insulator plastic rubber", "Insulators resist the flow of electricity."),

    # Heat and temperature
    ("temperature heat thermal", "Temperature measures how hot or cold something is."),
    ("heat transfer warm cold", "Heat moves from warmer objects to cooler objects."),
    ("conduction metal contact", "Conduction transfers heat through direct contact."),
    ("convection liquid air movement", "Convection transfers heat through moving fluids."),
    ("radiation sun heat waves", "Radiation transfers heat through electromagnetic waves."),
    ("celsius thermometer degrees", "Temperature is often measured in degrees Celsius (°C)."),

    # Waves and sound
    ("sound wave vibration frequency", "Sound comes from vibrations moving through a medium."),
    ("frequency hertz pitch", "Frequency is measured in Hertz (Hz)."),
    ("pitch high low sound", "Higher frequency creates higher pitch."),
    ("echo reflection sound", "An echo happens when sound reflects back."),
    ("wave wavelength amplitude", "Waves transfer energy without transferring matter."),

    # Light and optics
    ("light reflection mirrors optics", "Reflection happens when light bounces off a surface."),
    ("refraction glass water bend", "Refraction is the bending of light."),
    ("mirror image reflection", "Mirrors reflect light and form images."),
    ("lens magnify focus", "Lenses bend light to focus images."),
    ("transparent translucent opaque", "Transparent objects let light pass through; opaque objects do not."),

    # Matter and atoms
    ("atom matter molecule particle", "Atoms are tiny particles that make up matter."),
    ("electron proton neutron", "Atoms contain protons, neutrons, and electrons."),
    ("solid liquid gas states", "Matter exists mainly as solids, liquids, and gases."),
    ("density mass volume", "Density = mass / volume."),
    ("molecule chemical matter", "Molecules are groups of atoms bonded together."),
    ("evaporation boiling water", "Evaporation changes liquid into gas."),
    ("condensation water vapor", "Condensation changes gas into liquid."),
    ("melting freezing state", "Melting turns solid into liquid; freezing does the opposite."),

    # General school help
    ("physics lesson school science", "Which topic do you need help with? Forces, electricity, waves, or energy?"),
    ("exercise problem homework", "Send me your physics exercise and I'll help."),
    ("exam test study", "Focus on formulas, concepts, and solving practice problems."),
    ("thanks thank you appreciate", "You're welcome! Feel free to ask more physics questions.")
]

question_texts = [q for q, a in qa_pairs]
question_embeddings = model.encode(question_texts, convert_to_tensor=True)

# Similarity threshold — below this, the bot says it doesn't understand
THRESHOLD = 0.3


# Semantic matching replaces keyword matching
def get_response(user_input):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(input_embedding, question_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    if best_score < THRESHOLD:
        return best_score, "Sorry, I don't understand. Try asking about sound, atoms, or atom matter!"

    return best_score, qa_pairs[best_idx][1]

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Physics Chatbot")
        self.root.geometry("500x600")
        self.root.configure(bg="#2E2E2E")

        # Title
        tk.Label(
            root, text="Physics Chatbot", font=("Helvetica", 16, "bold"),
            fg="#FFFFFF", bg="#2E2E2E"
        ).pack(pady=10)

        # Chat area (scrollable)
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, height=20, width=50, font=("Arial", 11),
            bg="#3C3C3C", fg="#E0E0E0", insertbackground="white"
        )
        self.chat_area.pack(pady=10, padx=10)
        self.chat_area.insert(tk.END,
                              "Welcome to the Physics Chatbot!\n"
                              "Ask about Physics.\n")
        self.chat_area.config(state='disabled')

        # Input frame
        input_frame = tk.Frame(root, bg="#2E2E2E")
        input_frame.pack(pady=5)

        # Input field
        self.input_field = tk.Entry(
            input_frame, width=40, font=("Arial", 11), bg="#4A4A4A", fg="#FFFFFF",
            insertbackground="white"
        )
        self.input_field.pack(side=tk.LEFT, padx=5)
        self.input_field.bind("<Return>", self.send_message)

        tk.Button(
            input_frame, text="Send", command=self.send_message, font=("Arial", 11),
            bg="#4CAF50", fg="#FFFFFF", activebackground="#45A049"
        ).pack(side=tk.LEFT, padx=5)

        # Clear button
        tk.Button(
            root, text="Clear Chat", command=self.clear_chat, font=("Arial", 11),
            bg="#F44336", fg="#FFFFFF", activebackground="#D32F2F"
        ).pack(pady=5)

    def send_message(self, event=None):
        user_input = self.input_field.get().strip()
        if not user_input:
            return

        score, response = get_response(user_input)
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"\nYou: {user_input}\n")
        self.chat_area.insert(tk.END, f"Match confidence: {score: .2f}\n")
        self.chat_area.insert(tk.END, f"Bot: {response}\n")
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)
        self.input_field.delete(0, tk.END)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.insert(tk.END,
                              "Welcome to the Physics Chatbot!\n"
                              'ask about Physics.\n')
        self.chat_area.config(state='disabled')

def main():
    root =tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()