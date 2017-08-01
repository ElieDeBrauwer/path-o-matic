import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# define coefficients
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

#define the features and label
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

#define the model
#linear_model = W*x + b
model = tf.nn.softmax(tf.matmul(x,W) + b, name="model")

prediction = tf.argmax(model, 1, name="prediction")
loss = -tf.reduce_sum(y * tf.log(model))

#pick the training algorithm
optimizer= tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#define the accuracy
correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("done: features, label, model and algorithm")

# create a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#set training and test data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MINST_data', one_hot=True)

model_saver = tf.train.Saver()
TRAIN_STEPS = 1000
for i in range(TRAIN_STEPS+1):
    x_train, y_train = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x:x_train, y:y_train})
        
# Train the model and save it in the end
model_saver.save(sess, "saved_models/Images.ckpt")
print("done: saving model")
