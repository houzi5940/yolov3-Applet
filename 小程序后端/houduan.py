from flask import Flask, request, jsonify
import json
from common_util import *
from yolo import *
import os

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# static_dir = os.path.join(BASE_DIR, 'static')
app = Flask(__name__, static_folder='static')


@app.route('/shabiyouxuan', methods=["GET", "POST"])
def get_imagedata():
    yolo = YOLO()
    data = request.files
    file = data.get("file")
    file.save("./data/1.jpg")

    print('**********************************')
    img = './data/1.jpg'
    # image = open(img, 'rb').read()
    image = Image.open(img)
    a = yolo.detect_image(image)
    a.save('result.png')
    image_data = open('result.png', "rb").read()
    img_stream = base64.b64encode(image_data)
    s = img_stream.decode()
    return jsonify({'ok': s})


@app.route('/shabihouzi', methods=["GET", "POST"])
def get_videodata():
    data = request.files
    file = data.get("file")
    file.save("./1.mp4")
    img = './1.mp4'
    yolo = YOLO()
    detect_video(yolo, img, 'static/result.mp4')
    print('**********************************')
    video = 'static/result.mp4'
    return json.dumps({'data': video})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8808')
