# 100 Days Of Code - Log

### Day 0: 20 June 2020

**Today's Progress**: Planned what I am going to do for the challenge - working through the content and tutorials in [i]Mastering Machine Learning for Penetration Testing[/i] by Chiheb Chebbi ([Kindle link:](https://read.amazon.com.au/kp/kshare?asin=B07CSN7QQ1&id=QHSWWvxaREGnhO8MOKuj1g&reshareId=NWJ5D150EP9SWGK4X9J2&reshareChannel=system). Also began reading the introductory parts. 

**Thoughts:** There is a lot of material and a lot of topics covered - I'm especially interested in the malware detection and automated analysis techniques.

**Link to work:** See link to Kindle book above. I searched through a bunch of resources, reviewed my notes re: automated malware analysis from the malware reverse engineering course I did in my Masters degree and read through the introductory parts of the above book. I also made sure I had CUDA, Python and everything else installed on my PC. Didn't write any actual code today, but got everything ready for me to do so.

### Day 1: 21 June 2020

**Today's Progress**: Today, I finished configuring my environment and the various packages needed, including Tensorflow, Keras, Matplotlib, Theano, Numpy, nltk, scikit-learn, pandas. I made sure they all worked and configured them.

**Thoughts:** The book was written a few years ago, so some instructions and code samples are out of date. This is especially the case with the TensorFlow documentation. However, I eventually did get everything working after some Google & Github searches and by reading the TensorFlow docs, including with GPU integration for TF.

**Link to work:** Jupyter Notebook input and output showing tests of packages:

```python
# Test Tensorflow
# NOTE: This one required searches because the book used a different version of Python and TensorFlow, so the original instructions in the book threw an exception. I needed to use TensorFlow v1 compatibility versions of the functions.
import tensorflow as tf
with tf.compat.v1.Session() as sess:
    # Build a graph
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    
    # Evaulate the tensor to show the result
    print(sess.run(c))
    
    # Print a constant message
    Message = tf.constant("Hello, world")
    print(sess.run(Message))
```

    30.0
    b'Hello, world'
    


```python
# Numpy version
import numpy
print (numpy.__version__)
```

    1.18.1
    


```python
# Keras version
import keras
print (keras.__version__)
```

    2.3.1
    

    Using TensorFlow backend.
    


```python
# Test pandas Series function
import pandas
data = numpy.array(['m','a','t','e',' ','m','a','l','i','c','e'])
SR = pandas.Series(data)
print(SR)
```

    0     m
    1     a
    2     t
    3     e
    4      
    5     m
    6     a
    7     l
    8     i
    9     c
    10    e
    dtype: object
    


```python
# Test matplotlib
import matplotlib.pyplot as plt
x = numpy.linspace(0, 20, 50)
plt.plot(x, x, label='linear')
plt.legend()
plt.show()
```


![png](output_4_0.png)



```python
# Test Theano
from theano import *
import theano.tensor as T
from theano import function
a = T.dscalar('a')
b = T.dscalar('b')
c = a + b
f = function([a,b],c)
print(f)
```

    <theano.compile.function_module.Function object at 0x000002AFF02A8D08>
    


```python
# Download NLTK packages
import nltk
nltk.download()
# This didn't work in a Jupyter Notebook, though when run from the command line, it did open the window prompt for downloading the NLTK packages.
```
