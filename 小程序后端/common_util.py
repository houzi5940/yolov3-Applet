import numpy as np
import cv2
import base64

RESULTS = {
    200: {
        'code': 200,
        'msg': 'success'
    },
    400: {
        'code': 400,
        'msg': 'require paremeter'
    }
}

def get_result_model():
    return {'code': '', 'msg': '', 'data': {}, 'cost_time': 0}

def base64_2_array(base64_data):
    im_data = base64.b64decode(base64_data)
    im_array = np.frombuffer(im_data, np.uint8)
    im_array = cv2.imdecode(im_array, cv2.COLOR_BAYER_BG2BGR)
    return im_array

def base64_1_array(base64_data):
    im_data = base64.b64encode(base64_data)
    im_array = np.frombuffer(im_data, np.uint8)
    im_array = cv2.imdecode(im_array, cv2.COLOR_BAYER_BG2BGR)
    return im_array

def result_data(data, cost_time=0):
    result = get_result_model()
    result['msg'] = RESULTS[200]['msg']
    result['code'] = RESULTS[200]['code']
    result['data'] = data
    result['cost_time'] = cost_time
    return result