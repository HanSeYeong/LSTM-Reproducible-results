# LSTM-Reproducible-results
How to get reproducible result when running Keras with Tensorflow backend

----------------------------------------------------------------------------------------------------------------------
##### When you are running your code with CPU, the 'Failed 2' case will solve your reproducible problem.
##### But the real problem is running with GPU.
##### Because of the randomness in GPU proceeding, we must handle multi threads with one thread and limit the CUDNN using.
##### I've tried so many codes to make my LSTM model can produce same results.
##### And Finally I got the answer! So I want to share the way with a clear documet!
----------------------------------------------------------------------------------------------------------------------

### Failed 1
Using theano backend
I've changed backend with theano.

.keras/keras.jason ->
```
{
  "epsilon": 1e-07,
  "floatx": "float32",
  "backend": "tensorflow",    <---- change to "theano"
  "image_data_format": "channels_last"
}
```
and implement python with below code. ( in my case, I used mnist_cnn.py )
```
THEANO_FLAGS="dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic" python mnist_cnn.py
```

but the results are all different every trying.

----------------------------------------------------------------------------------------------------------------------

### Failed 2
set numpy random seed
```
random_seed=2017 
from numpy.random import seed 
seed(random_seed)
```
----------------------------------------------------------------------------------------------------------------------

### Failed 3
```
set tensorflow random seed

from tensorflow import set_random_seed
set_random_seed(random_seed)
```
----------------------------------------------------------------------------------------------------------------------

### Failed 4
```
set build-in random seed

import random
random.seed(random_seed)
```
----------------------------------------------------------------------------------------------------------------------

### Failed 5
```
set PYTHONHASHSEED

import os
os.environ['PYTHONHASHSEED'] = '0'
```
----------------------------------------------------------------------------------------------------------------------

### Solve the randomness of LSTM results.
```
from __future__ import print_function
from numpy.random import seed
import random
import tensorflow as tf

import os
os.environ['PYTHONHASHSEED'] = '0'
seed(42)
random.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
```
and write your codes which you just wrote.
in my case~
```
model = Sequential()
model.add(LSTM(rnn_width, stateful=False, return_sequences=True,
                        input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(rnn_dropout))
model.add(AttentionDecoder(rnn_width, n_lotto_numbers))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```
----------------------------------------------------------------------------------------------------------------------
**Cautions** : You must follow the order of above code.
We must set numpy.random.seed before import keras and the order of the other codes are same!
My writings are not exactly right, because I learned above informations in stackflows, git questions...
I hope this code will help your reproducible problem!
