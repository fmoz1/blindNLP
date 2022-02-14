# Deep NN
## Intro
1. Modern techniques (algorithms) 
  * Batch size
  * Momentum
  * Choosing hyperparameters: hidden layers, hidden units, learning rate, etc.  
  * Weight initialization: some rule of thumns 
  * Dropout regularization 
2. Modern libraries and hardware (GPUs)
  * Theano 
  * TensorFlow 
  * GPU (cloud, purchase own hardware)
  
## Review
1. Logistic regression ("neuron")
   * Predictions 
   * Loss function
2. Neural Networks
   * Predictions
   * Backpropagation
3. Regularization and Overfitting  

### Classification
1. K output classes:
   $$ p(t=k|x) = \frac{a_k}{\sum_{j=1}^K exp(a_j)} $$ 
2. Define loss function L
   * Binary: binary crossentropy (negative log Bernoulli likelihood)
   * Multiclass: categorical crossentropy 

## Neuron learning/training
1. How to minimize the loss function? 
   * Past: dL/dW = 0 dL/db = 0 
   * NN: Repeat $$\theta_{new} = \theta_{old} - \eta \nabla_{\theta_{old}} L $$ where $\eta$ is the learning rate. 
   * Note: always plot the loss per iteration.
2. What are the gradients?
  $$ \frac{\delta L}{\delta W_{ik}} = \sum (y_{nk} - t_{nk}) x_{ni} $$
  In vectorized form, 
  $$ \nabla_W L = x^T (Y-T)$$
  * X: N x D
  * Y: N x K
  * Targets: N x K 
  $$ \nabla_b L = (Y-T)^T 1 $$
  * 1: vector of ones N x 1 
## ANN 
1. NN: logistic regression stacked together 
2. Input to hidden
   $$ a_1 = W_1^T x + b_1 $$
   $$ z = \sigma(a_1) $$ 
   Note: $a_1$ and z are vectors of size M. ReLU is more commonly used than sigmoid in practice. 
3. Hidden t i output
   $$a_2 = W_2^T z + b_2 $$ 
   $$y  = softmax(a2) $$ 
   Note: $a_2$ and y are vectors of size K. 
4. Generalization and overfitting 
   * Regularization penalty: $$\lambda ||\theta||_2^2 $$ ($\lambda$ is a hyperparameter)

## Stochastic Gradient Descent
1. Full GD


       for i in range(num_epochs):
           gradient = grad(X, Y, params)
           params = params - learning_rate * gradient


2. Mini-batch


        for i in range(num_epochs):
            shuffle(X,Y)
            for Xb, Yb in get_batches(X,Y):
                gradient = grad(Xb, Yb, params)
                params = params - learning_rate * gradient


3. Implementation 

     
     num_batches = int(np.ceil(N/B))
     for j in range(num_batches):
        Xb = X[j* B : (j+1) * B]
        Yb = Y[j * B : (j+1) * B]


4. Note
   * Typical B = 8, 16, 32, 64, ... 
   * Why do we want large B? 
   * Larger B = more accurate
   * Smaller B = more iteration (SGD is slower than other gradient descent)