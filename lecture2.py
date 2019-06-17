import tensorflow as tf
import os
import numpy as np

"""
For speed up CPU computations
"""
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(x))

# Warning?
# The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(x))


"""
Visualize with TensorBoard
"""
## Visualize it with TensorBoard

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph()) # Create the summary writer after graph definition and before running your session


with tf.Session() as sess:
	print(sess.run(x))

writer.close() # close the writer when you’re done using it

# Go to terminal, then run (on your working dir):
# $ tensorboard --logdir="./graphs" --port any port number you want
# Then open your brower and go to : http://localhost:6666

## Explicitly name variables
# You can check the variables name are "Constant x" on the tensorboard using name='variable name'

a = tf.constant(2, name='a') 
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	print(sess.run(x)) # >> 5


"""
Constants, Sequences, Variables, Ops
"""
# Constants
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.multiply(a, b, name='mul1')

with tf.Session() as sess:
    print(sess.run(x)) # >> [[0 2][4 6]]

# Tensors filled with a specific value
zeros = tf.zeros([2, 3], tf.int32)                      # Fill zeros(dtype=int32) in tensor which has [2, 3] shape 
zeros_like = tf.zeros_like([[0, 1], [5, 4], [2, 3]])    # Fill zeros in a tensor. The size of the tensor is same as input tensor
fill_specific_value = tf.fill([2, 3], 8)                # Fill specific value (8) in specific size of tensor ([2, 3])

with tf.Session() as sess:
    print(sess.run(zeros))                 # [[0 0 0][0 0 0]]
    print(sess.run(zeros_like))            # [[0 0][0 0][0 0]]
    print(sess.run(fill_specific_value))   # [[8, 8, 8][8, 8, 8]]

# TF vs NP Data Types
tf.int32 == np.int32           # True, 
a = tf.zeros([2, 3], np.int32) # Can pass numpy types to TensorFlow operations
print(type(a))                 # >> <class 'tensorflow.python.framework.ops.Tensor'>

# Create variable with operation outputs
with tf.Session() as sess:
    a = sess.run(a) 

print(type(a)) # >> <class 'numpy.ndarray'>

# It is recommended to avoding these problem as below:
with tf.Session() as sess:
    a_out = sess.run(a) 

print(type(a))       # >> <class 'tensorflow.python.framework.ops.Tensor'>
print(type(a_out))   # >> <class 'numpy.ndarray'>

###############################################################################################################
# Q) What's wrong with constants?                                                                             #      
# A) Constants are stored in the graph definition. This makes loading graphs expensive when constants are big.#
# Use Variables or readers instead of.                                                                        #                      
# tf.Constant : operator, tf.Variable : class                                                                 #             
###############################################################################################################

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

tf.reset_default_graph()
# Variables need to be initialized
# There will be error because Variables are not initialized
with tf.Session() as sess:
	print(sess.run(W))

# The easiest way is initializing all variables at once:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(W.eval())     # Similar to print(sess.run(W))
    print(sess.run(W)) 
    
# Initialize only a subset of variables:
with tf.Session() as sess:
	sess.run(tf.variables_initializer([s, m]))

# Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

# tf.Variable.assign()
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print("Before assign : ", W.eval())
    sess.run(assign_op)                 
    print("After assign : ", W.eval())  # Variable.assign is an operator and the operator needs to be executed in a session.

# Each session maintains its own copy of variables
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) 		# >> 20
print(sess2.run(W.assign_sub(2))) 		# >> 8

print(sess1.run(W.assign_add(100))) 	# >> 120
print(sess2.run(W.assign_sub(50))) 		# >> -42

sess1.close()
sess2.close()

"""
Placeholder
o Empty place for variables
o TensorFlow create a graph before execute operations 
o And we want to create the graph without specific values
o You can put specific values into a placeholder with "feed_dict"
"""
# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(c, feed_dict={a: [1, 2, 3]})) # >> [6, 7, 8]

    