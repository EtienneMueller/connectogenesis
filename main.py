import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


tf.keras.mixed_precision.set_global_policy('mixed_float16')


def load_coordinate(file_path, layer_shift = 16):
    # here: 1212 time steps = 10:06 minutes
    # here: length ~7900 neurons in plane 15 etc
    length = len(np.load(file_path + f"/plane{0+layer_shift}/stat.npy", allow_pickle=True))
    x_array = np.empty(length)
    y_array = np.empty(length)
    z_array = np.empty(length)
    brightness = np.random.rand(length, 1212)
    
    for i in range(1):
        stat_file = file_path + f"/plane{i+layer_shift}/stat.npy"
        stat = np.load(stat_file, allow_pickle=True)

        f_file = file_path + f"/plane{i+layer_shift}/F.npy"
        f = np.load(f_file, allow_pickle=True)
        
        for j in range(length):
            x_array[j] = stat[j]['med'][0]
            y_array[j] = stat[j]['med'][1]
            z_array[j] = i
            brightness[j] = f[j]

    b_min = np.min(brightness)
    b_max = np.max(brightness)
    brightness = np.divide(brightness-b_min, b_max-b_min)
    x, y, z, brightness = filter(x_array, y_array, z_array, brightness)

    return x, y, z, brightness


def load_timesteps(file_path):
    for i in range(50):
        ts = np.load(file_path + f"/plane0/stat.npy", allow_pickle=True)
    print(len(ts))
    

def filter(x, y, z, brightness, brightness_threshold=0.32):
    average_brightness = np.mean(brightness, axis=1)
    mask = average_brightness >= brightness_threshold
    x, y, z = x[mask], y[mask], z[mask]
    
    brightness = brightness[mask]
    
    b_min = np.min(brightness)
    b_max = np.max(brightness)
    
    brightness = np.divide(brightness-b_min, b_max-b_min)

    return x, y, z, brightness


def load_and_stack_data(data_folder="data/suite2p", num_planes=50):
    stacked_data = []
    
    for i in range(num_planes):
        plane_folder = os.path.join(data_folder, f"plane{i}")
        file_path = os.path.join(plane_folder, "F.npy")
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            stacked_data.append(data)
            #print(i, data.shape)
        else:
            print(f"File not found: {file_path}")
    
    # Stack all arrays vertically
    combined_array = np.vstack(stacked_data).T
    print("loaded raw data")

    #data = np.random.rand(100, 50)  # Example data with 100 data points and 50 time steps
    # Define input (X) and shifted output (y)
    x = combined_array[:-1]  # Use all rows (time steps) and all columns (data points) as input
    y = np.zeros((1211, 9508))  # Initialize y with the same shape as data
    y[:, 0:9508] = combined_array[1:, 0:9508]  # Shifted top half for target variable

    # Expand dimensions if necessary to make the data 3D (samples, timesteps, features)
    # x = np.expand_dims(x, axis=-1)
    # y = np.expand_dims(y, axis=-1)

    print(f"{x.shape = }, {y.shape = }")

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"{x_train.shape = }, {x_test.shape = }, {y_train.shape = }, {y_test.shape = }")
    print("created train/test set")

    return x_train, x_test, y_train, y_test 


def save_to_tfrecord(data, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for array in data:
            feature = {
                'data': tf.train.Feature(float_list=tf.train.FloatList(value=array.flatten()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"{file_path} saved to tfrecord")


def parse_tfrecord(example_proto):
    feature_description = {'data': tf.io.FixedLenFeature([605], tf.float32)}
    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example['data']


# Function to load TFRecord files with batching
def load_tfrecord(file_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset.batch(batch_size)


def nn(filepath):
    batch_size = 1
    x_train = load_tfrecord('data/x_train.tfrecord', batch_size)
    y_train = load_tfrecord('data/y_train.tfrecord', batch_size)

    # data = tf.data.TFRecordDataset(filepath)
    # data = data.batch(20).prefetch(tf.data.AUTOTUNE) 

    # Define a simple model with a direct connection between input and output
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(605,), dtype=tf.float16),
        tf.keras.layers.Flatten(),  # Flatten to ensure direct mapping
        # tf.keras.layers.Flatten(shape=(401693,)),  # Flatten to ensure direct mapping
        tf.keras.layers.Dense(9508, activation='linear')  # Direct connection to output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mae')

    # Train the model
    train_dataset = tf.data.Dataset.zip((x_train, y_train)).prefetch(tf.data.AUTOTUNE)
    history = model.fit(train_dataset, epochs=10, batch_size=16) #, validation_data=(X_test, y_test))

    # Evaluate the model
    #loss = model.evaluate(x_test, y_test)
    #print("Test loss:", loss)


def nn_batch(x_train, x_test, y_train, y_test):
    # Assuming `data` and `labels` are your large NumPy arrays
    # data = np.array(...)  # Replace with your actual data
    # labels = np.array(...)

    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(1).prefetch(1)  # Use batch size of 1 and prefetch to improve performance

    # Define your model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),  # Adjust input shape
        # tf.keras.layers.Dense(units=10000, activation='relu'),  # Example layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=9508, activation='linear') # , activation='linear')  # Example output
    ])

    # Compile and train the model
    print("compiling")
    model.compile(optimizer='adam', loss='mae')  # Adjust as needed
    print("start training")
    model.fit(dataset, epochs=10, verbose=1)  # Adjust epochs as needed


def main():

    if not os.path.exists('data/x_train.tfrecord'):
        x_train, x_test, y_train, y_test  = load_and_stack_data()
        save_to_tfrecord(x_train, 'data/x_train.tfrecord')
        save_to_tfrecord(x_test, 'data/x_test.tfrecord')
        save_to_tfrecord(y_train, 'data/y_train.tfrecord')
        save_to_tfrecord(y_test, 'data/y_test.tfrecord')
        # numpy_to_tfrecord(data, 'data/data.tfrecord')

    # nn('data/')

    x_train, x_test, y_train, y_test = load_and_stack_data()
    nn_batch(x_train, x_test, y_train, y_test)

    # x, y, z, brightness = load_coordinate('data/suite2p')
    # print(x, y, z, brightness)
    # load_timesteps('data/suite2p')


if __name__=='__main__':
    main()
