import tensorflow as tf

matrixA = tf.constant([[2,0],[4,2]])
matrixB = tf.constant([[0,6],[18,3]])

if ((matrixA.shape[2] == 0)
    matrixA = tf.expand_dims(matrixA,1)
else if ((matrixB.shape[2] == 0)
    matrixB = tf.expand_dims(matrixB,1)

resultado = tf.matmul(matrixA,matrixB)

print(tf.Session().run(resultado))

