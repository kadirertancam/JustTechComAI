import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

MODEL_NAME = "openai/gpt-oss-20b"
HF_TOKEN = os.getenv("HF_TOKEN") 

# Hugging Face'e giriş yap
login(token=HF_TOKEN)

# Model ve tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    token=HF_TOKEN
)

# Text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95
)

# Gradio arayüzü için fonksiyon
def chat_fn(history, user_input):
    prompt = user_input
    response = generator(prompt)[0]["generated_text"]
    # Basitçe, kullanıcının prompt'unu ve modeli yanıtını kaydet
    history = history + [(user_input, response)]
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# gpt-oss-20B Chat")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Prompt", placeholder="Merhaba...")
    state = gr.State([])

    def submit_fn(user_message, history):
        history, _ = chat_fn(history, user_message)
        return history, history

    msg.submit(submit_fn, [msg, state], [chatbot, state])

# Web arayüzü
demo.launch(server_name="0.0.0.0", server_port=7860)
