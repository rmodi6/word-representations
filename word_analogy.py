import os
import pickle
import numpy as np


model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""


def cossim(v1, v2):
    '''
    Returns the cosine similarity between two vectors A and B defined as A.B/(||A||*||B||)
    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity of v1 and v2
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def difference(word1, word2):
    '''
    Returns the difference vector between two words
    :param word1: first word
    :param word2: second word
    :return: vector(word1) - vector(word2)
    '''
    v1 = embeddings[dictionary[word1]]
    v2 = embeddings[dictionary[word2]]
    return np.subtract(v1, v2)

with open('word_analogy_dev_predictions_{}.txt'.format(loss_model), 'w') as fw:
    with open('word_analogy_dev.txt', 'r') as fr:
        for line in fr:  # For each line in the file
            examples, choices = line.strip().split('||')  # Separate examples and choices
            examples = examples.strip('"').split('","')  # Remove quotes and separators to create a list of examples
            choices = choices.strip('"').split('","')  # Remove quotes and separators to create a list of choices
            total_diff = 0
            for example in examples:  # For each example calculate the difference between vectors of the words
                words = example.split(':')
                total_diff += difference(words[1], words[0])
            avg_diff = total_diff / len(examples)  # Calculate the average difference for all examples
            similarities = []
            for choice in choices:  # For each choice
                words = choice.split(':')
                diff = difference(words[1], words[0])  # Calculate the difference between vectors of the words
                similarities.append(cossim(diff, avg_diff))  # Store the similarity of difference with avg difference
            most_illustrative = choices[similarities.index(max(similarities))]  # Find most similar choice
            least_illustrative = choices[similarities.index(min(similarities))]  # Find least similar choice
            choices.extend([most_illustrative, least_illustrative])
            write_line = '"{0}"\n'.format('"\t"'.join(choices))
            fw.write(write_line)  # Write the choices along with least and most similar choice to file
