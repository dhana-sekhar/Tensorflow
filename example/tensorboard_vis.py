import tensorflow as tf
X = tf.placeholder(tf.float32, shape=[3], name = "X")  
Y = tf.Variable([2.0, 4.0, 6.0], tf.float32)
Z = tf.constant([1, 2, 3], tf.float32)

res = X * Y + Z
#sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    X_val = [3, 6, 9]
    sess.run(init)
    
    tf.summary.FileWriter('./graphs', sess.graph)

    result = sess.run(res, feed_dict={X:X_val})
    Y_val = Y.eval()
    Z_val = Z.eval()

    print(X_val, Y_val, Z_val, result)