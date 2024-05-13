from flask import Flask, request, jsonify, render_template, send_file
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
import torch
import io
import os

app = Flask(__name__)

# Directorio de checkpoints
#checkpoint_dir = r"C:\Users\sergi\Desktop\SDGUI\Models\Checkpoints"

# Función para cargar el modelo desde un checkpoint
#def load_model_from_checkpoint(model):
#    modelT = torch.load(model)
#    return modelT
    

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data['prompt']
    num_inference_steps = int(data['num_inference_steps'])  # Convertir a entero
    model = data['model']
    CFG = float(data['CFG'])
    #checkpoint_path = os.path.join(checkpoint_dir, model)
    #torchModel = load_model_from_checkpoint(model)
    # Generar la imagen usando el modelo cargado desde el checkpoint
    pipeline = StableDiffusionPipeline.from_single_file(model)
    pipeline.to(torch_device="cuda", torch_dtype=torch.float32)
    images = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=CFG, lcm_origin_steps=50, output_type="pil").images
    # Convertir la imagen a bytes
    img_byte_array = io.BytesIO()
    images[0].save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    # Devolver la imagen generada
    return send_file(img_byte_array, mimetype='image/png')

@app.route('/')
def index():
    return render_template('inicio.html')

@app.route('/libro_de_vida')
def libro_de_vida():
    return render_template('ejhtml.html')

@app.route('/generar_imagen')
def generar_imagen():
    return render_template('index.html')

# Nueva ruta para listar los checkpoints disponibles
@app.route('/list_checkpoints')
def list_checkpoints():
    checkpoints = os.listdir(checkpoint_dir)
    return jsonify(checkpoints)

if __name__ == '__main__':
    app.run(debug=True)
