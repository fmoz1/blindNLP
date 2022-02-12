# Notes on Advanced NLP 
## Intro 
1. Bidirectional RNN 
2. Seq2seq 
   *  Applications: machine translations, doesn't require input length to be equal to output length
3. Attention mechanism 
   * On top on seq2seq  
4. Memory networks 
   * Question answering 
   * Machine intelligence 
  
## Review: RNN, CNN
### Summary 
1. CNN
   * CNN most seen for images, RNN is the focus for texts 
   * CNNs are simpler, since they don't consider time, don't need recurrent connections, also they are faster 
2. RNN
   * All ML interfaces are the same  
  
### Word embedding
1. NLP: machine learning applied to text/speech 
2. Feature vector (X1, X2, X3, ..., Xn)
   * Word vectors: features vectors correspond to words (e.g., king: (gender, age, place))
3. How to find word vectors?
   * Unsupervised learning: They only make sense geometrically, and are like hidden layers that don't make sense to humans
   * Word2Vec, GloVE, FastText 
4. Word similarity 
5. Matrix representation: V x D (vocab size x feature vector size)
   * V = # of rows = vocab size = # distinct words
   * D = embedding dimension = feature vector size 
   * Every dense layer is: f(input.dot(weights) + bias)
   * A word embedding is simply a matrix of stacked word vectors 
   * You should NEVER multiple a one-hot vector by a matrix -> inefficient
   * TF: tf.nn.embedding_lookup()

### Using word embeddings 
1. Pretrained embedding: W = preloaded from csv 
2. Can initialize these uncommon words to random (or zero)
3. Fine-tune or not:
   * embedding_layer.trainable = True (False)
4. CNN: Convolution is better called correlation. 

## RNN
1. Input: x(t) = given (shape D)
2. Hidden layer: h(t) = f(W_i x(t) + Wh h(t-1) + b_h) (shape M)
3. Dense: y(t) = softmax(W0 h(t) + b0) (shape K)
Note: shape(W_i) = D x M, shape(W_h) = M x M, shape(W_0) = M x K

### RNN vs. feedforward neural networks
1. Feedforward example
   * T = 20 (seq length), D = 10 (input dim), M = 15 (hidden layer size), K = 3 (output classes)
   * Input to hidden weights of size : T x D x T x M = 60,000 
   * Hidden to output weights of size: T x M x T x K = 18,000 // 78,000 total weights!
2. RNN 
   * Input to hidden: 10 x 15 = 150, hidden to hidden: 15 x 15 = 225, hidden to output: 15 x 3 = 45 // 420 total weights
3. Feedback NN cons:
   * Many parameters
   * They must have a constant size, and yet real sequences often have unequal size 
4. Dealing with variable length sequences
   * Benefits of constant size 
     * RNN cells in Keras and TF require constant-length sequences 
     * Data is always a rectangularly shaped N x T x D 
     * scan() can be confusing 
   * Cons: we have to choose T 
   
### GRU
1. Gates allow you to "remember" or "forget" values.
   
Update gate:
$$ z_t = \sigma(W_{xz}^T x_t + W_{hz}^T h_{t-1} + b_z) $$
Reset gate:
$$ r_t = \sigma(W_{xr}^T x+t W_{hr}^T h_{t-1} + b_r) $$
Candidate state: 
$$ \hat{h_t} = tanh(W_{xh}^T x_t + W_{hh}^T (r_t \times h_{t-1}) + b_h $$
Next state:
$$ h_t = (1-z_{t}) \times h_{t-1} + z_t \times \hat{h_t}$$
Note:
* $z_t$ is unique to GRU. Weight to new $\hat{h_t}$
* Update, reset gates and candidate statr are "mini neural networks"
* Concat trick 
  * we can combine $X_t$ and $h_{t-1}$ into $[X_t, h_{t-1}]$

### LSTM 
1. Invented in 1997 
2. Output two things: h and c 
3. A gate for everything 
   
Forget gate:
$$ f_t = \sigma(W_f^T[X_t, h_{t-1}] + b_f) $$

Input gate:
$$ i_t = \sigma(W_i^T[X_t, h_{t-1}] + b_f) $$ 

Output gate:
$$ o_t = \sigma(W_o^T[X_t, h_{t-1}] + b_f) $$ 

Candidate cell: 

$$\tilde{c}_t = tanh(W_c [X_t, h_{t-1}] + b_c) $$

Cell state (or the other hidden state)
$$ c_t = f_t \times c_{t-1} + i_t \times \tilde{c_t} $$

Hidden state:
$$ h_t = o_t \times tanh(c_t) $$

1. Keras implementation
   * For recurrent units, we can return hidden states: return_state = True
   * Return all the h's? return_sequences = True 


### Think shapes 
1. Input shape: a sequence of vectors T x D (e.g., weather 24 hours x 10 locs)
2. Output shape: T x K
3. Other NLP applications
   * POS tagging: T predictions, instead of 1 (like spam/ham)
   * Machine translation: input and output length differ ($T_x, T_y$)

4. Apply RNN to images 
   * H x W = 2 dimensions; can regard H as T, W as D ß
   * We can also rotate the image and run bidrectional RNN on both, and so go in all 4 directions
   * Bi-LSTM latent dim = M what does output look like in shape? 
     * N x H x W -> N x H x 2M -> N x 2M (output)
     * By rotation: N x W x H -> N x W x 2M -> N x 2M (output)
     * Concat 2 N x 2M to N x 4M 
     * Dense layer -> N x K

## Sequence-to-sequence 
### Architecture
1. Dual RNN
   * Encoder: no outputs, because not making predictions; only keep $h_T$ (return_sequences = False), the final state with static size M (thought vector)
   * Decoder
2. The decode
   * New RNN unit
   * Must have same vector size M 
   * Pass <start-of-sentence> token into the "x" input
3. Implementation issues
   * Keras works with constant-sized data 
   * Decoder input length during training is $T_y$
   * Decode input length during prediction is 1 (conflict!)
   * **Teacher forcing** 
     * Pass in the true target sequence 
     * Why? Difficult to learn to generate the entire sentence at once
     * If you get a word wrong, the teacher may correct you, allowing you to work off the corrected sentence so far
     * Note: must be offset by <SOS>
4. Solution 
   * Create 2 different models for training and sampling.
   * Pseudo-code:

   `emb = Embedding(); lstm = LSTM(); dense = Dense()`

   `input1 = Input(length = Ty)`
   `model1 = Model(input1, dense(lstm(emb(input1)))`

   `input2 = Input(length = 1`

   `model2 = Model(input2, dense(lstm(emb(input2)))`

   `h = encoder model output; x= <SOS>`

   `for t in range(Ty):` 

   `x, h = model2.predict(x,h)`

### Language modeling
1. Next word prediction:  
   $$P(w_t | w_{t-1}, w_{t-2}, ...)$$ 