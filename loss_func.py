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
    ############# Initialization #############
    # Initialize required variables
    batch_size = inputs.shape[0]
    neg_samples = tf.convert_to_tensor(sample)  # [k, 1]
    unigram_probs = tf.convert_to_tensor(unigram_prob)  # [vocab, 1]
    k = float(sample.shape[0])  # number of negative samples
    noise = float(10 ** -10)  # small number to avoid log(0)

    # Initialize target words variables
    target_weights = tf.nn.embedding_lookup(weights, tf.reshape(labels, [batch_size]))  # [batch_size, embed_size]
    target_biases = tf.nn.embedding_lookup(biases, labels)  # [batch_size, 1]
    target_unigram_probs = tf.nn.embedding_lookup(unigram_probs, labels)  # [batch_size, 1]

    # Initialize negative words variables
    sample_weights = tf.nn.embedding_lookup(weights, neg_samples)  # [k, embed_size]
    sample_biases = tf.nn.embedding_lookup(biases, tf.reshape(sample, [-1, 1]))  # [k, 1]
    sample_unigram_probs = tf.nn.embedding_lookup(unigram_probs, tf.reshape(sample, [-1, 1]))  # [k, 1]

    ############# Calculations with center and target words #############
    # Calculate dot product of center word u_c with target word u_o : (uT_c u_o)
    dot_product_1 = tf.matmul(target_weights, inputs, transpose_b=True)  # [batch_size, batch_size]
    self_dot_prod_1 = tf.linalg.diag_part(dot_product_1)  # [batch_size,]
    self_dot_prod_1 = tf.reshape(self_dot_prod_1, [-1, 1])  # [batch_size, 1]
    # Add bias : s(w_o , w_c ) = (uT_c u_o) + b_o
    s1 = tf.add(self_dot_prod_1, target_biases)  # [batch_size, 1]

    # Calculate unigram prob of target word and take log : log [kPr(w_o)]
    p1 = tf.log(tf.add(tf.scalar_mul(k, target_unigram_probs), noise))  # [batch_size, 1]

    # x = s(w_o , w_c ) - log [kPr(w_o)]
    x1 = tf.subtract(s1, p1)  # [batch_size, 1]
    # Calculate prob of word pair in vocab : Pr(D = 1, w_o |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_1 = tf.sigmoid(x1)  # [batch_size, 1]
    lhs = tf.log(tf.add(noise, sigma_1))  # [batch_size, 1]

    ############# Calculations with center and negative sample words #############

    # Calculate dot product of center word u_c with negative sample words u_x : (uT_c u_x)
    dot_product_2 = tf.matmul(sample_weights, inputs, transpose_b=True)  # [k, batch_size]
    # Add bias : s(w_x , w_c ) = (uT_c u_x) + b_x
    s2 = tf.add(dot_product_2, sample_biases)  # [k, batch_size]

    # Calculate unigram prob of negative words and take log : log [kPr(w_x)]
    p2 = tf.log(tf.add(tf.scalar_mul(k, sample_unigram_probs), noise))  # [k, 1]

    # x = s(w_x , w_c ) - log [kPr(w_x)]
    x2 = tf.subtract(s2, p2)  # [k, batch_size]
    # Calculate prob of word pair in vocab : Pr(D = 1, w_x |w_c ) = sigma(x) = 1 / (1 + e^(-x))
    sigma_2 = tf.sigmoid(x2)  # [k, batch_size]
    # Calculate prob of word pair not being in vocab and take log : log(1 - Pr(D = 1, w_x |w c ))
    rhs = tf.reduce_sum(tf.log(tf.add(noise, tf.subtract(1., sigma_2))), axis=0)  # [1, batch_size]
    rhs = tf.reshape(rhs, [-1, 1])  # [batch_size, 1]


    # Calculate final loss for the batch
    j = tf.negative(tf.add(lhs, rhs))  # [batch_size, 1]
    return j