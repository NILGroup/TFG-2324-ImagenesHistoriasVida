from flask import Flask, request, jsonify, render_template, send_file
from diffusers import DiffusionPipeline
import torch
import io

app = Flask(__name__)

# Carga el modelo y configura los dispositivos
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe.to(torch_device="cuda", torch_dtype=torch.float32)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data['prompt']
    num_inference_steps = int(data['num_inference_steps'])  # Convertir a entero

    # Generar la imagen
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images

    # Convertir la imagen a bytes
    img_byte_array = io.BytesIO()
    images[0].save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Devolver la imagen generada
    return send_file(img_byte_array, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
