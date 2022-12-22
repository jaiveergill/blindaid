import load_model as utils
from tensorflow.keras.models import load_model
from keras.applications.xception import Xception
import pickle

class Model:
    def __init__(self, model_location="resources/model_9.h5", tokenizer="resources/tokenizer.p") -> None:
        self.past_outputs = []
        self.model = load_model(model_location)
        self.tokenizer=pickle.load(open(tokenizer,"rb"))
        self.xception_model = Xception(include_top=False, pooling="avg")

    def return_output(self, img_path):
        return utils.scene_desc(self.model, self.tokenizer, img_path)