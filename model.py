import tensorflow as tf

def model():
  g = tf.Graph()
  with g.as_default():
    in_image_tensor = tf.placeholder(tf.uint8, shape=[1, 10, 10, 3], name='image_tensor')
    tf.constant([3.0], name='num_detections')
    tf.constant([[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]], name='detection_boxes')
    tf.constant([[0.1, 0.2, 0.3]], name='detection_scores')
    tf.identity(tf.constant([[1.0, 2.0, 3.0]]) * tf.reduce_sum(tf.cast(in_image_tensor, dtype=tf.float32)), name='detection_classes')
    graph_def = g.as_graph_def()
  with tf.gfile.Open("model.pb", 'wb') as f:
    f.write(graph_def.SerializeToString())

if __name__ == "__main__":
    model()
