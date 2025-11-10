from flask import Flask, request, jsonify, send_from_directory, current_app
import os
import atexit
from werkzeug.utils import secure_filename
from flask_cors import CORS
from main import SID
import inference_models.SAFE_Series.inference_SR as SAFE_RINE 
import inference_models.SAFE_Series.inference_SAFE as SAFE
import acl


app = Flask(__name__)


CORS(app)
app.config['UPLOAD_FOLDER'] = 'images'
app.config["AUDIO_FOLDER"] = "audios"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# 香橙派后端服务器ip和端口
ip_addr = '10.181.236.140'
port = 4000

@app.route('/audios/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect_endpoint():
    #acl.rt.set_context(current_app.context)  # ⬅️ 每次请求必须加这一行！
    # 从应用上下文中获取detector
    detector = current_app.detector
    
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    filename = secure_filename(file.filename)
    filepath = os.path.join(cur_dir, app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    timestamp = request.form.get('timestamp', 'unknown')
    print(f"✅ Image received: {filepath}, time:{timestamp}")
    
    
    # 调用检测函数
    acl.rt.set_context(detector.acl_resource.context)
    result = SID(filepath, model = app.model, explain=True, speak=True, detector=detector, timestamp = timestamp, mode = "web")
    
    # 构建响应
    return jsonify({
        'result': result['category'],
        'confidence': result['confidence'],
        'explanation': result['explanation'],
        'audioUrl': f"http://{ip_addr}:{port}/audios/{result['audio_filename']}"
    })

    
if __name__ == '__main__':

    model = "SAFE" # choose model
    app.model = model
    print(f"Initialize detector...")
    if model == "SAFE":
       app.detector = SAFE.SAFE_Init()  # save detector as app property
       atexit.register(lambda: SAFE.SAFE_DeInit(app.detector)) # register DeInit function to clear resource
    elif model == "SAFE_RINE":
       app.detector = SAFE_RINE.SR_Init()   
       atexit.register(lambda: SAFE_RINE.SR_DeInit(app.detector))
    
    print(f"🌍 Server running: http://{ip_addr}:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
    
    