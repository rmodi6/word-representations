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
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def difference(word1, word2):
    v1 = embeddings[dictionary[word1]]
    v2 = embeddings[dictionary[word2]]
    return np.subtract(v1, v2)

with open('word_analogy_dev_predictions_{}.txt'.format(loss_model), 'w') as fw:
    with open('word_analogy_dev.txt', 'r') as fr:
        for line in fr:
            examples, choices = line.strip().split('||')
            examples = examples.strip('"').split('","')
            choices = choices.strip('"').split('","')
            total_diff = 0
            for example in examples:
                words = example.split(':')
                total_diff += difference(words[1], words[0])
            avg_diff = total_diff / len(examples)
            similarities = []
            for choice in choices:
                words = choice.split(':')
                diff = difference(words[1], words[0])
                similarities.append(cossim(diff, avg_diff))
            least_illustrative = choices[similarities.index(min(similarities))]
            most_illustrative = choices[similarities.index(max(similarities))]
            choices.extend([least_illustrative, most_illustrative])
            write_line = '"{0}"\n'.format('"\t"'.join(choices))
            fw.write(write_line)
