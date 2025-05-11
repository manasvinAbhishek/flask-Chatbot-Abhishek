from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None

@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history_ids
    user_input = ""
    bot_response = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        # Encode user input and add end of string token
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Append input to chat history or start a new one
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Create attention mask
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

        # Decode last generated tokens
        bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return render_template("index.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
