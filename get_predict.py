import tensorflow as tf

import re
import string

from tensorflow.keras import layers


class ModelPredict():

    def __init__(self) -> None:

        max_features = 10000
        sequence_length = 250

        vectorize_layer = layers.TextVectorization(
                            standardize=self.custom_standart,
                            max_tokens=max_features,
                            output_mode='int',
                            output_sequence_length=sequence_length)
        model = None
        

    def custom_standart(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, '[%s]' %re.escape(string.punctuation), '')

    

    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label



    def load_model(self):
       self.model = tf.keras.models.load_model('./saved_model/my_model/',
                                    custom_objects={'custom_standart': self.custom_standart, 
                                                    'vectorize_text': self.vectorize_text})
        

    def get_predict(self, rewiev: list):

        predict = self.model.predict(rewiev)

        return predict
    