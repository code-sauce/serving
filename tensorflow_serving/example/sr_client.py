import requests
from bottle import request, route, run
import json
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


IMAGE_CLASSIFICATION_SERVER_HOST = '0.0.0.0'
IMAGE_CLASSIFICATION_SERVER_PORT = 9000
MODEL_NAME = 'model.ckpt-145000'


def _get_image_content(image_url):
    """
    Given an image URL, return the image contents (bytes)
    """
    response = requests.get(image_url)
    return response.content


@route('/classify')
def classify_image():
    image_url = request.GET.get('image_url')
    image_content = _get_image_content(image_url)
    prediction_request = predict_pb2.PredictRequest()
    prediction_request.model_spec.name = MODEL_NAME
    prediction_request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image_content, shape=[1]))
    channel = implementations.insecure_channel(IMAGE_CLASSIFICATION_SERVER_HOST, int(IMAGE_CLASSIFICATION_SERVER_PORT))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Predict(prediction_request, 10.0)  # 10 secs timeout
    return json.dumps({'data': str(result.outputs.values())})


@route('/health')
def health_check():
    return json.dumps({'OK': 1})

run(host='0.0.0.0', port=8001)
