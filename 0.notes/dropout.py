keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# keep_prob allows you to adjust the number of units to drop
# in order to compensate for dropped units, 
# tf.nn.dropout() multiplies all units that are kept by 1/keep_prob

# train: start with keep_prob = 0.5
# test: keep_prob = 1










