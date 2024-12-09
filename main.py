from transformers import pipeline
import gradio as gr

# Modelo de geração de texto (usando GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Função para gerar texto com base no input do usuário
def generate_text(prompt, max_length, num_return_sequences):
    results = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return "\n\n".join([result["generated_text"] for result in results])

# Interface Gradio
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Digite o início do texto...", label="Prompt de Texto"),
        gr.Slider(10, 200, step=10, value=50, label="Comprimento Máximo do Texto"),
        gr.Slider(1, 5, step=1, value=1, label="Número de Sequências Geradas"),
    ],
    outputs=gr.Textbox(label="Texto Gerado"),
    title="Gerador de Texto com GPT-2",
    description="Digite um prompt para gerar texto com base no modelo GPT-2."
)

# Lançar a aplicação
interface.launch()
