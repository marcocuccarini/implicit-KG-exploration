import ollama

class OllamaChat:
    def __init__(self, model):
        self.model = model

    def send_prompt(self, prompt):
        """
        Sends a prompt to Ollama. Named 'send_prompt' to match 
        the calls in the iterative_explanation logic.
        """
        try:
            r = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"Ollama Error: {e}")
            return None

    # Optional: keep 'ask' as an alias just in case
    def ask(self, prompt):
        return self.send_prompt(prompt)