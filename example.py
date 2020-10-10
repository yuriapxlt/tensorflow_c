import tensorflow as tf

def example():
    a = tf.placeholder(tf.float32, shape=(1, 100), name="input_a")
    b = tf.placeholder(tf.float32, shape=(1, 100), name="input_b")
    c = tf.add(a, b, name='result')
    i = tf.initializers.global_variables()
    with open('session.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

if __name__ == "__main__":
    example()
