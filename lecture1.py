import tensorflow as tf

"""
Tensorflow does not directly compute an operator
They create the "graph" and then compute
"""

# We expect the result 8(3+5) as below because we use add method
# However tensorflow just create graph

a = tf.add(3, 5)
print(a) # type : int32

b = tf.add(3.0, 5.0)
print(b) # type : float32


# How to get the value of a?
# Create a session, assign it to variable sess
# Intuitively create a graph with tensorflow methods and run (evaluate) the graph using session

a = tf.add(3, 5)        # create a graph which for add 3 and 5
sess = tf.Session()     # call Session to compute the graph
print(sess.run(a))      # print output
sess.close()            # close Session

# with clause takes care of sess.close()
with tf.Session() as sess:
    print(sess.run(a))      # this is a python syntax


# More complex situation

# 1
x, y = 2, 3
op1 = tf.add(x, y)             # create add node in a graph
op2 = tf.multiply(x, y)        # create multiplication node in the graph
op3 = tf.pow(op2, op1)         # create pow node in the graph
with tf.Session() as sess:
    op3 = sess.run(op3)        # evaluate the graph created in above lines

# 2
x, y = 2, 3
op1 = tf.add(x, y)               # create add node in a graph
op2 = tf.multiply(x, y)          # create multiplication node in the graph
useless = tf.multiply(x, op1)    # create multiplication node in the graph (not used in sess)
op3 = tf.pow(op2, op1)           # create pow node in the graph
with tf.Session() as sess:
    op3 = sess.run(op3)          # evaluate the graph created in above lines
                                 # In this case, useless node is not active in compute process

# 3
x, y = 2, 3
op1 = tf.add(x, y)                                  # create add node in a graph
op2 = tf.multiply(x, y)                             # create multiplication node in the graph
useless = tf.multiply(x, op1)                       # create multiplication node in the graph 
op3 = tf.pow(op2, op1)                              # create pow node in the graph
with tf.Session() as sess:
    op3, not_useless = sess.run([op3, useless])     # evaluate the graph created in above lines
                                                  

# Distributed computation
# To put part of a graph on a specific CPU or GPU

# Creates a graph
with tf.device('/GPU:0'):    # set device as the first GPU, '/cpu:0' for using cpu
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True
# To show log
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))

#######################################################################################################################
# Create a graph
g = tf.Graph()
with g.as_default():    # set a default graph
    a, b = 3, 5
    x = tf.add(a, b)

with tf.Session(graph=g) as sess:   #
    print(sess.run(x)) # return 8

# Reset a graph
tf.reset_default_graph()

"""
Do not mix default graph and user created graphs
"""
# Below lines are not recommended
g = tf.Graph()

# add operators to the default graph
a = tf.constant(3)

# add operators to the user created graph
with g.as_default():
    b = tf.constant(5)


# It is better than above lines
g1 = tf.get_default_graph()
g2 = tf.Graph()

# add operators to the default graph
with g1.as_default():
    a = tf.constant(3)

# add operators to the user created graph
with g2.as_default():
    b = tf.constant(5)


#######################################################################################################################
# Tensorflow create graph and compute sub-graph when user call session
# User can assign each other device such as the first GPU, second GPU or CPU on each sub-graph computation
# It is the reason tensorflow create a graph first of all
#######################################################################################################################