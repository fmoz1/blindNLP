# CNN
## Convolution aka cross-correlation
1. Two requirements: 
   * Can you add?
   * Can you multiply?
2. Terminology 
   * Input (image)
  
          $*$ (element multiplication)
         
   * Filter (kernel) 
   * Examples: blurring, edge detection 
   * The filter decides how specific convolution works.
3. Pseudo-code: 
   * Scanning image with filter
     * Input l (w) = N
     * Kernel (w) = K
     * Output l (w) = N - K +1
   * Notes: images may not be squares, kernels are usually squares. 


      output_height = input_height - kernel_height +1

      output_width = input_width - kernel_width +1 

      output_images = np.zeros([output_height, output_width])

      for i in range(0, output_height):

         for j in range(0, output_width):

           for ii in range(0, kernel_height):

            for jj in range(0, kernel_width):` 

              output_image[i,j] += input_image[i+ii, j+jj] * kernel[ii, jj]
   * Convolution equation in deep learning is actually cross-coreelation. 
   $$ (A * w)_{ij} = \sum_{i'=0}^{K-1} \sum_{j'=0}^{K-1}  A(i+i', j+j')w(i', j')$$
   It doesn't matter. Convolution is commutative. 

   True convolution is instead: 
      $$ (A * w)_{ij} = \sum_{i'=0}^{K-1} \sum_{j'=0}^{K-1}  A(i-i', j-j')w(i', j')$$
   * Implementation 
   
   `convolve2d(A, np.fliplr(np.flipud(w)), mode = 'valid)`

     * The movement of the filter is bounded by the edges of the image. The output is always smaller than the input image. 
     * If one wants input size = output size, use padding. 
     * Summary of modes:
        * Valid: output size N-K+1 
        * Same: N
        * Full: N+K-1 (atypical) 

## Intuition for CNN
1. Vectorization  
   $$ a^T b = \sum_{i=1}^N a_i b_i $$ (element wise sum and add)

   $$ a \cdot b = \sum_{i=1}^N a_i b_i = |a| |b| cos \theta_{ab}$$ (cosine distance)

2. 1-D convolution 
   Example: 
   $$ a = (a_1, ..., a_4), 
   w = (w_1, w_2) $$

   $$ b = a * w = (a_1 w_1 + a_2 w_2, a_2 w_1 + a_3 w_2, a_3 w_1 + a_4 w_2) $$

   Equivalently, 
   $$ b_i = \sum_{i' = 1}^K a_{i+i'} w_{i'} $$

   Equivalently, a matrix multiplication in the form:

   `np.array([[w1, w2,0, 0], [0, w1, w2, 0], [0, 0, w1, w2]]) * (np.array([a1, a2, a3, a4])).T `

   Note: w is of size 2. The matrix is 3 x 4. Takes too much space. 

3. Parameter sharing / weight sharing 
   $$ a = W^T x $$
   where W is the weight matrix. 


4. Convolution on color images 
   * 3 dimension dot product
   * This breaks uniformity, because input is 3D but output is 2D 
   * Use multiple filters
   * Input: H x W x 3 => H x W x (arbitrary #) which is called "feature map" not more as color 

### CNN Architecture 

1. Convolutional layer vs. dense layer
   * Conv layer = pattern finder 
     * Max/average pooling: less data to process; translational invariance (*) (position \
   doesn't matter for recognizing an A)
     * Different pool sizes: pooling has flexibility (boxes can overlap "stride")
     * CNN gradually loses spatial information (doesn't care where the feature was found)
      \ but gain information about what features were found 

2. Hyperparameters 
   * So many choices
   * Learning rate, # hidden layers, # hidden units per layer
   * Convention with CNN: small filters, repeat convolution -> pooling, increase # of feature maps 32 > 64 > 128

3. Stride 
   * Strided convolution, or conv + pooling 
   * Feature maps increase with each layer

4. Dense 
   * Dense layer expects a 1-D input vector (use `flatten()`)
   * Global max pooling: H x W x C -> 1 x 1 x C 
   * Keras does this automatically: doesn't care where the feature was found 

5. Summary
   * Step 1: Conv > Pool > Conv > Pool ..
   * Step 2: Flatten() / GlobalMaxPolling2D()
   * Step 3: Dense > Dense 

6. Functional API method for building CNN 