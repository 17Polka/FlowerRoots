from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем модель
try:
    model = YOLO('best.pt')
    logger.info("✅ Модель загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки: {e}")
    model = None

CHESS_SQUARE_SIZE_MM = 10

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>FloraRoots · анализ растений</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        h1 i { color: #4CAF50; }
        .status {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-size: 18px;
        }
        .upload-area {
            border: 3px dashed #4CAF50;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .upload-area i {
            font-size: 64px;
            color: #4CAF50;
            margin-bottom: 15px;
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover { background: #45a049; }
        .result {
            display: none;
            margin-top: 30px;
        }
        .result img {
            max-width: 100%;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .metric-card {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .metric-card i {
            font-size: 30px;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .unit { color: #999; font-size: 14px; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            border-radius: 10px;
            width: 0%;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><i class="fas fa-leaf"></i> FloraRoots</h1>
        
        <div class="status">
            <i class="fas fa-check-circle" style="color: #4CAF50;"></i>
            Сервер работает 24/7 · Hugging Face 🤗
        </div>

        <div class="upload-area" id="dropArea">
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Нажмите или перетащите фото</p>
            <input type="file" id="fileInput" accept="image/*" hidden>
            <button class="btn" id="selectBtn">Выбрать фото</button>
            <button class="btn" id="cameraBtn">Снять на камеру</button>
        </div>

        <div class="progress-bar" id="progressBar">
            <div class="progress-fill" id="progressFill"></div>
        </div>

        <div class="result" id="result">
            <img src="" alt="Результат" id="resultImage">
            <div class="metrics" id="metrics"></div>
        </div>
    </div>

    <script src="https://kit.fontawesome.com/yourcode.js" crossorigin="anonymous"></script>
    <script>
        const dropArea = document.getElementById('dropArea');
        const fileInput = document.getElementById('fileInput');
        const selectBtn = document.getElementById('selectBtn');
        const cameraBtn = document.getElementById('cameraBtn');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const result = document.getElementById('result');
        const resultImage = document.getElementById('resultImage');
        const metrics = document.getElementById('metrics');

        selectBtn.onclick = (e) => {
            e.stopPropagation();
            fileInput.click();
        };

        cameraBtn.onclick = (e) => {
            e.stopPropagation();
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.capture = 'environment';
            input.onchange = () => {
                if (input.files[0]) analyze(input.files[0]);
            };
            input.click();
        };

        dropArea.onclick = () => fileInput.click();

        dropArea.ondragover = (e) => {
            e.preventDefault();
            dropArea.style.background = '#f0fff0';
        };

        dropArea.ondragleave = () => {
            dropArea.style.background = 'white';
        };

        dropArea.ondrop = (e) => {
            e.preventDefault();
            dropArea.style.background = 'white';
            if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
        };

        fileInput.onchange = () => {
            if (fileInput.files[0]) analyze(fileInput.files[0]);
        };

        async function analyze(file) {
            const formData = new FormData();
            formData.append('image', file);

            // Preview
            const reader = new FileReader();
            reader.onload = (e) => resultImage.src = e.target.result;
            reader.readAsDataURL(file);

            progressBar.style.display = 'block';
            result.style.display = 'none';
            progressFill.style.width = '30%';

            try {
                progressFill.style.width = '60%';
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                progressFill.style.width = '90%';
                const data = await response.json();

                if (data.measurements) {
                    metrics.innerHTML = '';
                    
                    const organs = [
                        { name: 'Корень', icon: 'root', data: data.measurements.root },
                        { name: 'Стебель', icon: 'tree', data: data.measurements.stem },
                        { name: 'Листья', icon: 'leaf', data: data.measurements.leaf }
                    ];

                    organs.forEach(org => {
                        if (org.data && org.data.count > 0) {
                            metrics.innerHTML += `
                                <div class="metric-card">
                                    <i class="fas fa-${org.icon}"></i>
                                    <h3>${org.name}</h3>
                                    <div>
                                        <span class="value">${org.data.total_length.toFixed(1)}</span>
                                        <span class="unit">мм</span>
                                    </div>
                                    <div>
                                        <span class="value">${org.data.total_area.toFixed(1)}</span>
                                        <span class="unit">мм²</span>
                                    </div>
                                </div>
                            `;
                        }
                    });
                }

                if (data.image) {
                    resultImage.src = 'data:image/jpeg;base64,' + data.image;
                }

                progressFill.style.width = '100%';
                setTimeout(() => {
                    progressBar.style.display = 'none';
                    result.style.display = 'block';
                }, 500);

            } catch (error) {
                alert('Ошибка: ' + error.message);
                progressBar.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

def analyze_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)[0]
    
    measurements = {
        'root': {'total_length': 0, 'total_area': 0, 'count': 0},
        'stem': {'total_length': 0, 'total_area': 0, 'count': 0},
        'leaf': {'total_length': 0, 'total_area': 0, 'count': 0}
    }
    
    if results.masks is not None:
        for mask, cls in zip(results.masks.data, results.boxes.cls):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            
            area_pixels = np.sum(mask_np)
            mm_per_pixel = 0.05  # примерный коэффициент
            area_mm2 = area_pixels * (mm_per_pixel ** 2)
            
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(contour)
                length_pixels = max(rect[1])
                length_mm = length_pixels * mm_per_pixel
            else:
                length_mm = 0
            
            class_name = results.names[int(cls)].lower()
            if class_name in measurements:
                measurements[class_name]['count'] += 1
                measurements[class_name]['total_length'] += length_mm
                measurements[class_name]['total_area'] += area_mm2
    
    img_with_masks = results.plot()
    _, buffer = cv2.imencode('.jpg', img_with_masks)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {'measurements': measurements, 'image': img_base64}

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'Нет изображения'}), 400
    
    try:
        result = analyze_image(request.files['image'].read())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)