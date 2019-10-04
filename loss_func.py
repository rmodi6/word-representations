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
    # Calculate dot product of center word v_c with all predicting words u_w
    dot_product = tf.matmul(true_w, inputs, transpose_b=True)  # [batch_size, batch_size]

    # Extract dot product of center word v_c with its target word u_o
    self_dot_prod = tf.linalg.diag_part(dot_product)  # [1, batch_size]
    # Calculate numerator A
    A = tf.log(tf.exp(self_dot_prod))  # [1, batch_size]

    # Summation of dot products for center word v_c with all predicting words u_w
    sum_of_exps = tf.reduce_sum(tf.exp(dot_product), axis=1)  # [1, batch_size]
    # Calculate denominator B
    B = tf.log(sum_of_exps)  # [1, batch_size]

    # Return the difference -(A-B) = B-A
    # no need to reshape this to [batch_size, 1] as reduce_mean will calculate mean across both axis
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
    k = float(sample.shape[0])
    noise = float(10 ** -10)

    preds_o = tf.nn.embedding_lookup(weights, tf.reshape(labels, [batch_size]))  # [batch_size, embed_size]
    biases_o = tf.nn.embedding_lookup(biases, labels)  # [batch_size, 1]
    probs_o = tf.nn.embedding_lookup(unigram_prob_tensor, labels)  # [batch_size, 1]

    preds_x = tf.nn.embedding_lookup(weights, samples_tensor)  # [k, embed_size]
    biases_x = tf.nn.embedding_lookup(biases, tf.reshape(sample, [-1, 1]))  # [k, 1]
    probs_x = tf.nn.embedding_lookup(unigram_prob_tensor, tf.reshape(sample, [-1, 1]))  # [k, 1]

    dot_product_1 = tf.matmul(preds_o, inputs, transpose_b=True)  # [batch_size, batch_size]
    self_dot_prod_1 = tf.linalg.diag_part(dot_product_1)  # [batch_size,]
    self_dot_prod_1 = tf.reshape(self_dot_prod_1, [-1, 1])  # [batch_size, 1]
    # Calculate s(w_o , w_c ) = (uT_c u_o) + b_o
    s1 = tf.add(self_dot_prod_1, biases_o)  # [batch_size, 1]
    # log [kPr(w_o)]
    p1 = tf.math.log(tf.add(noise, tf.scalar_mul(k, probs_o)))  # [batch_size, 1]
    # x = s(w_o , w_c ) - log [kPr(w_o)]
    x1 = tf.subtract(s1, p1)  # [batch_size, 1]
    tf.nn.nce_loss
    # sigma_1 = tf.divide(1., tf.add(1., tf.exp(tf.math.negative(x1))))  # Pr(D = 1, w_o |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_1 = tf.sigmoid(x1)  # [batch_size, 1]
    lhs = tf.math.log(tf.add(noise, sigma_1))  # [batch_size, 1]

    #########################################

    dot_product_2 = tf.matmul(preds_x, inputs, transpose_b=True)  # [k, batch_size]
    # s(w_x , w_c ) = (uT_c u_x) + b_x
    s2 = tf.add(dot_product_2, biases_x)  # [k, batch_size]
    # log [kPr(w_x)]
    p2 = tf.math.log(tf.add(noise, tf.scalar_mul(k, probs_x)))  # [k, 1]

    # x = s(w_x , w_c ) - log [kPr(w_x)]
    x2 = tf.subtract(s2, p2)  # [k, batch_size]
    # sigma_2 = tf.divide(1., tf.add(1., tf.exp(tf.math.negative(x2))))  # Pr(D = 1, w_x |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_2 = tf.sigmoid(x2)  # [k, batch_size]
    rhs = tf.reduce_sum(tf.math.log(tf.add(noise, tf.subtract(1., sigma_2))), axis=0)  # [1, batch_size]
    rhs = tf.reshape(rhs, [-1, 1])  # [batch_size, 1]

    j = tf.math.negative(tf.add(lhs, rhs))
    return j