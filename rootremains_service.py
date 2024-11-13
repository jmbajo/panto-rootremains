from flask import Flask, request, jsonify
import os
import uuid
import __main__
from flask_cors import CORS, cross_origin


# Path to model checkpoint
checkpoint_file = './lib/model.pkl'

# Folder with the images to test
img_folder = './test-imgs/root_remains_negative'

# List with the name of the classes to predict (in order)
labels = ['root_remains_negative', 'root_remains_positive']


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

from fastai.vision.all import *

# Class that allows the use of Albumentations
class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

__main__.AlbumentationsTransform = AlbumentationsTransform

from lib.inference_fastai import FastAIClassifierInferencer


@app.route('/', methods=['GET'])
@cross_origin()
def test():
    return "ok-root-remains-service"


@app.route('/process_panto', methods=['POST'])
@cross_origin()
def procesar_imagen():
    data = request.get_json()
    if data is None:
        return "Data Vacía"
    
    if "image_uuid" not in data or "imgs_path" not in data:
        return jsonify({'mensaje': 'Faltan datos'})

    image_uuid = data["image_uuid"]
    imgs_path = data["imgs_path"]

    # print(image_uuid, imgs_path)

    imgs = get_image_files(imgs_path)
    # print(imgs)
    pos_list = [path.stem for path in imgs]
    print(pos_list)

    inferencer = FastAIClassifierInferencer(checkpoint_file, labels)
    inferencer.init(cpu=True)
    results = inferencer.process(imgs)

    to_ret = {}
    for i in range(len(pos_list)):
        to_ret[pos_list[i]] = results[i]

    print(results)
    return to_ret


if __name__ == '__main__':
    # Asegúrate de que la carpeta_local exista antes de ejecutar el script
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    app.run(host="0.0.0.0", port=8525)
