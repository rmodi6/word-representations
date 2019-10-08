## CSE 538: Natural Language Processing - Assignment 1 

**Name:** Ruchit Modi  
**SBU ID:** 112685342

**Note:** All the word2vec models are trained using tensorflow-gpu=1.14 and python=2.7 while all the nce models are 
trained using tensorflow=1.9 and python=2.7

**Models Link:** https://drive.google.com/drive/folders/1O6qngIvG7vPjg25XO9pwNsKeLXTiPPCw?usp=sharing

**Best Models:** 
- **Cross Entropy:** The best model for cross entropy loss was generated using an embedding size of 64 and training 
the model from scratch for 4 million steps without loading the pre-trained vectors.
- **NCE:** The best model for nce loss was created by increasing the window size. The value for skip_window was 8 and 
that for num_skips was 16. The training was continued from the pre-trained model for 200000 steps with all other 
parameters same.

### Implementations
#### Batch Generation
The basic algorithm to generate the batches is such that the global variable `data_index` keeps track of the last word 
used for batch generation and also always points to the center word for the current batch. It starts from `skip_window` 
position as we need `skip_window` words on left hand side of center word for a batch and ends at 
`len(data) - skip_window` as we need `skip_window` words on right hand side of center word. It is re-initialized back 
to start if it reaches the end of data.  
For every center word the data index points to, `num_skips` batches are generated by iterating `skip_window` times 
towards left and right side of `data_index`. Once the window is completed, `data_index` is incremented by 1 to make the 
next word center word and the loop continues until a batch of `batch_size` is completed.

#### Cross Entropy Loss
The calculation for cross entropy loss requires two variables A and B where A is the numerator part of the log 
likelihood function while B is the denominator part.  
For A, we need to find the dot product of center word with its target word for all words in the batch. This can be done 
by matrix multiplication of the `inputs` and `true_w` matrix and then taking only the diagonal part of the matrix. 
Then take the log and exp of the resultant (which is not required as they both cancel out each other) to get A.
For B, we need to find the dot product of center word with all other target words in vocab but for simplicity we only 
calculate the dot product with all other target words in current batch only. We can reuse the matrix multiplication 
used for A and calculate exps of each value in matrix. Use reduce sum to sum over all other target words and then 
calculate the log to get the value of B.  
Return the value of -(A-B)

#### NCE Loss
The calculation of nce loss can be divided into two parts - one involving the center word and target word, and the 
other involving center word and negative words. Using the same matrix multiplication technique used in cross entropy, 
calculate the dot product of a center word with target word and add bias. Then subtract the log unigram probability of 
k times the target word from it and take the sigmoid to get the probability of a center,target word pair being in vocab. 
Similarly, calculate the probability of center,negative word pair being in vocab and subtract it from 1 to get the 
probability of the pair not being in vocab. To do this for each negative word in the current negative sample, we can 
use matrix multiplication and then use reduce sum to find the summation. Finally, calculate the difference between the 
probability of center,target word pair being in vocab and the probability of center,negative word pair not being in 
vocab and return its negative for Gradient Descent.

#### Word Analogy
To calculate the least and most similar choice for a relation in a given set of examples, calculate the average 
difference of vectors between the words of each example and find the choice whose difference vector is least and most 
similar to the average vector. For similarity, we can use the cosine similarity of vectors which is calculated as 
`similarity = A.B/(||A||*||B||)`


### Experiments
1. **Embedding Size:** Changed the `embedding_size` from 128 to **64** and commented out loading of the pre trained 
model for both cross entropy loss and nce loss. Increased `max_num_steps` to 4000000 as it is being trained from 
scratch. All other parameters are same.
2. **Batch Size:** Changed `batch_size` from 128 to **256** for both cross entropy and nce loss keeping all other 
parameters same.
3. **Window Size:** Changed `skip_window` from 4 to **8** and `num_skips` from 8 to **16** for both cross entropy and 
nce loss keeping all other parameters same.
4. **Learning Rate:** Changed `learning_rate` from 1 to **1.5** for both cross entropy and nce loss keeping all other 
parameters same.
5. 1. **Num Steps:** Changed `max_num_steps` from 200001 to **1000001** for cross entropy loss keeping all other 
    parameters same.
   2. **Num Negative Samples:** Changed `num_sampled` from 64 to **128** for nce loss keeping all other parameters same.