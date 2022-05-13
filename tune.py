!pip install scikit-optimize

from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import InputLayer, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import os

batch_size = 128
num_classes = 10
epochs = 2

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
else: # for Tensorflow, the format is [N_image, image_row, image_col, channel]
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize the pixels to [0, 1]
X_train /= 255
X_test /= 255

# subset 5000 samples from X_train as validation_set
X_train, X_val = X_train[0:55000,:,:,:], X_train[-5000:,:,:,:]
y_train, y_val = y_train[0:55000], y_train[-5000:]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_val.shape[0], 'validation samples')

y_train_vec = y_train
y_test_vec = y_test
y_val_vec = y_val
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)

def create_model(learning_rate, kernel_size, num_dense_layers):
    # Start construction of a Keras Sequential model.
    model = Sequential()
    model.add(InputLayer(input_shape=(img_rows, img_cols, 1, )))
    model.add(Conv2D(32, kernel_size=int(kernel_size),
                 activation='relu',
                 padding='same',
                 name = 'layer_conv_1'))
    model.add(Conv2D(64, kernel_size=int(kernel_size),
                 activation='relu',
                 padding='same',
                 name = 'layer_conv_2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        
        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(64,
                        activation='relu',
                        name=name))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(learning_rate),
              metrics=['accuracy'])    
    return model

def log_dir_name(learning_rate,
                 kernel_size,
                 num_dense_layers):
    # The dir-name for the TensorBoard log-dir.
    s = os.path.join(log_path, "lr_{0:.0e}_kernel_{1}_dense_{2}/")
    
    log_dir = s.format(learning_rate,
                       kernel_size,
                       num_dense_layers)
    return log_dir

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                             name='learning_rate')
dim_kernel_size = Integer(low=3, high=5, name='kernel_size')
dim_num_dense_layers = Integer(low=1, high=3, name='num_dense_layers')

dimensions = [dim_learning_rate, dim_kernel_size, dim_num_dense_layers]

log_path = './mnist_skopt_log'
if not os.path.exists(log_path):
    os.makedirs(log_path)
result_path = './mnist_skopt_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)
path_best_model = os.path.join(result_path+'best_model.keras')
best_accuracy = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, kernel_size, num_dense_layers):
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('kernel_size:', kernel_size)
    print('num_dense_layers:', num_dense_layers)
    print()
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate, kernel_size,  num_dense_layers)
     
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, kernel_size,  num_dense_layers)
     
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
    
    # Use Keras to train the model.
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[callback_log])
    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_accuracy'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()
    
    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

# initial hyperparameters for search
default_parameters = [1e-4, 3, 1]

# search by Gaussian Process Optimization
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=11, # min n_calls=11
                            x0=default_parameters)

# save the search result for future use
dump(search_result,result_path+'search_result.gz', compress=9)
# if you need to reload the optimization history, use the following command
#res_load = load(result_path+'search_result.gz')

# check the optimization result
plot_convergence(search_result)

# check the optimized solution
search_result.x

# Evaluate the performance of the best model on the test set
model_best = load_model(path_best_model)
score = model_best.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# check the optimization history by listing the combination of all x and 
# its corresponding funtion value
sorted(zip(search_result.func_vals, search_result.x_iters))
