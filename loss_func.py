import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    dot_product = tf.matmul(true_w, inputs, transpose_b=True)
    self_dot_prod = tf.linalg.diag_part(dot_product)
    A = tf.log(tf.exp(self_dot_prod))

    sum_of_exps = tf.reduce_sum(tf.exp(dot_product), axis=1)
    B = tf.log(sum_of_exps)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    batch_size = inputs.shape[0]
    samples_tensor = tf.convert_to_tensor(sample)  # [k, 1]
    unigram_prob_tensor = tf.convert_to_tensor(unigram_prob)  # [vocab, 1]
    noise = float(10 ** -10)

    preds_o = tf.nn.embedding_lookup(weights, tf.reshape(labels, [batch_size]))  # [batch_size, embed_size]
    biases_o = tf.nn.embedding_lookup(biases, labels)  # [batch_size, 1]
    probs_o = tf.nn.embedding_lookup(unigram_prob_tensor, labels)  # [batch_size, 1]

    preds_x = tf.nn.embedding_lookup(weights, samples_tensor)  # [k, embed_size]
    biases_x = tf.nn.embedding_lookup(biases, tf.reshape(sample, [-1, 1]))  # [k, 1]
    probs_x = tf.nn.embedding_lookup(unigram_prob_tensor, tf.reshape(sample, [-1, 1]))  # [k, 1]

    k = float(sample.shape[0])


    dot_product_1 = tf.matmul(inputs, preds_o, transpose_b=True)  # [batch_size, batch_size]
    self_dot_prod_1 = tf.linalg.diag_part(dot_product_1)
    self_dot_prod_1 = tf.reshape(self_dot_prod_1, [-1, 1])
    s1 = tf.add(self_dot_prod_1, biases_o)  # s(w_o , w_c ) = (uT_c u_o) + b_o
    p1 = tf.math.log(tf.add(noise, tf.scalar_mul(k, probs_o)))  # log [kPr(w_o)]

    x1 = tf.subtract(s1, p1)  # x = s(w_o , w_c ) - log [kPr(w_o)]
    # sigma_1 = tf.divide(1., tf.add(1., tf.exp(tf.math.negative(x1))))  # Pr(D = 1, w_o |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_1 = tf.sigmoid(x1)
    lhs = tf.math.log(tf.add(noise, sigma_1))

    #########################################

    dot_product_2 = tf.matmul(preds_x, inputs, transpose_b=True)  # [k, batch_size]
    # dot_product_2 = tf.reduce_sum(dot_product_2, axis=1)
    # dot_product_2 = tf.linalg.matrix_transpose(dot_product_2)
    s2 = tf.add(dot_product_2, biases_x)  # s(w_x , w_c ) = (uT_c u_x) + b_x
    p2 = tf.math.log(tf.add(noise, tf.scalar_mul(k, probs_x)))  # log [kPr(w_x)]

    x2 = tf.subtract(s2, p2)  # x = s(w_x , w_c ) - log [kPr(w_x)]
    # sigma_2 = tf.divide(1., tf.add(1., tf.exp(tf.math.negative(x2))))  # Pr(D = 1, w_x |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_2 = tf.sigmoid(x2)
    rhs = tf.reduce_sum(tf.math.log(tf.add(noise, tf.subtract(1., sigma_2))), axis=0)
    rhs = tf.reshape(rhs, [-1, 1])

    j = tf.math.negative(tf.add(lhs, rhs))
    return j