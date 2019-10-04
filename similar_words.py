import os
import pickle
import numpy as np


model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

'''
Code to find 20 most similar words to the following words {first, american, would}
'''


def get_cosine_matrix(matrix, vector):
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    return np.divide(dotted, matrix_vector_norms)


words = ['first', 'american', 'would']
for word in words:
    word_embedding = embeddings[dictionary[word]]
    cosine_matrix = get_cosine_matrix(embeddings, word_embedding)
    top_20_indices = cosine_matrix.argsort()[-20:][::-1]
    top_20_words = [reverse_dictionary[i] for i in top_20_indices]
    print(word, top_20_words)
