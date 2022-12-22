import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.applications.xception import Xception, preprocess_input
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open("tokenizer.pickle","rb"))
model = load_model('model_9 (1).h5')
xception_model = Xception(include_top=False, pooling="avg")
MAX_LENGTH = 32

def extract_features(filename, model):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature


def gen_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        print(photo, photo.shape, sequence, sequence.shape)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
      if index == integer:
          return word
  return None

img_path = "images/img.jpg"
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)
description = gen_desc(model, tokenizer, photo, MAX_LENGTH)

def scene_desc(model, tokenizer, photo, max_length):
  description = gen_desc(model, tokenizer, photo, max_length)
  return description.replace("the", "a").replace("start", "").replace("end", "")