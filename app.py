from flask import Flask, request, jsonify
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

# ЗАГРУЖАЕМ НЕЙРОСЕТЬ
try:
    model = YOLO('best.pt')
    logger.info("✅ НЕЙРОСЕТЬ ЗАГРУЖЕНА!")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    model = None

# РАЗМЕР КЛЕТКИ (10 ММ)
CHESS_SQUARE_SIZE_MM = 10

def find_calibration(image):
    """Поиск шахматной доски для калибровки"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (4, 7)
    
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        pt1 = corners[0][0]
        pt2 = corners[1][0]
        pixel_dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        mm_per_pixel = CHESS_SQUARE_SIZE_MM / pixel_dist
        
        cv2.drawChessboardCorners(image, pattern_size, corners, found)
        return mm_per_pixel, image
    
    # Если доски нет - примерный коэффициент (для фото 2048x1536)
    default_mm = 0.05 * (2048 / image.shape[1])
    return default_mm, image

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'Нет изображения'}), 400
    
    file = request.files['image']
    try:
        # Читаем изображение
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Калибровка
        mm_per_pixel, img_with_chess = find_calibration(img.copy())
        
        # Анализ нейросетью
        results = model(img)[0]
        
        # Собираем результаты
        measurements = {
            'root': {'total_length': 0, 'total_area': 0, 'count': 0},
            'stem': {'total_length': 0, 'total_area': 0, 'count': 0},
            'leaf': {'total_length': 0, 'total_area': 0, 'count': 0},
            'flower': {'total_length': 0, 'total_area': 0, 'count': 0}
        }
        
        if results.masks is not None:
            for mask, cls in zip(results.masks.data, results.boxes.cls):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
                
                # Площадь
                area_pixels = np.sum(mask_np)
                area_mm2 = area_pixels * (mm_per_pixel ** 2)
                
                # Длина через контур
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
        
        # Рисуем результат
        img_with_masks = results.plot()
        
        # Добавляем текст
        cv2.putText(img_with_masks, f"1px = {mm_per_pixel:.4f}mm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img_with_masks)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info(f"✅ Анализ завершен: {measurements}")
        
        return jsonify({
            'measurements': measurements,
            'image': img_base64,
            'calibration': mm_per_pixel
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)