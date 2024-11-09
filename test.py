# Path to model checkpoint
checkpoint_file = './lib/model.pkl'

# Folder with the images to test
img_folder = './test-imgs/retained'

# List with the name of the classes to predict (in order)
labels = ['not_retained', 'retained']


from fastai.vision.all import *

# Class that allows the use of Albumentations
class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()

    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


from lib.inference_fastai import FastAIClassifierInferencer

imgs = get_image_files(img_folder)
print(imgs)


inferencer = FastAIClassifierInferencer(checkpoint_file, labels)
inferencer.init(cpu=True)
results = inferencer.process(imgs)

print(results)
