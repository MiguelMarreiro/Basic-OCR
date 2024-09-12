DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))


    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)

"""Usin KNN - k-nearest neighbors algorithm to implement OCR from scratch kNN is a supervised learning algorithm in
which 'k' represents the number of nearest neighbors considered in the classification or regression problem,
and 'NN' stands for the nearest neighbors to the number chosen for k."""

# Prep file paths
DATA_DIR = 'mnist_datasets/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels-idx1-ubyte'

N_TRAIN_IMAGES = 60000
N_TEST_IMAGES = 1

def bytes_to_int(byte_data):
    """bytes aren't iterable so we must convert to numbers """
    return int.from_bytes(byte_data, 'big')


# Read Image
def read_images(filename, n_max_images=None):
    """TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background(white),255 means foreground(black)."""
    images = []
    with open(filename, 'rb') as f:
        # we don't care about the magic number value as we already know the file type so we can assign it to a
        # placeholder _
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = min(n_images, n_max_images)
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for img_index in range(n_images):
            image = []
            for rows_index in range(n_rows):
                row = []
                for column_index in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    """TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9."""
    labels = []
    with open(filename, 'rb') as f:
        # we don't care about the magic number value as we already know the file type so we can assign it to a
        # placeholder _
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = min(n_labels, n_max_labels)
        for img_index in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def flatten_list(l):
    """returns a list of pixels from the 2D image sample"""
    return [pixel for sublist in l for pixel in sublist]


# Using list comprehension it creates a list of images as a 1d list of pixels
def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    return sum(
        [(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)]
    ) ** (0.5)

    # The zip(x, y) function in Python takes two (or more) iterables, x and y, and aggregates their elements into tuples


def get_training_distances_for_test_sample(X_train, test_sample):
    """we're returning the distance between our test digit and every training digit """
    return [dist(train_sample, test_sample) for train_sample in X_train]



def get_most_frequent(list):
    """
    chars = {}
    highest_count = 0
    most_frequent_char = ''
    for char in list:
        if char not in chars:
            chars[char] = 1
        else:
            chars[char] = chars[char] + 1
        if chars[char] > highest_count:
            highest_count = chars[char]
            most_frequent_char = char
    print(chars)
    print(most_frequent_char, highest_count)"""
    return max(list, key=list.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []  # predicted labels, our output

    """
    X_test = [image1, image2, ...]
    y_pred = [0,      9,      ...]
    """

    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[index]
            for index in sorted_distance_indices[:k]
            ]
        print(candidates)
        top_candidate = get_most_frequent(candidates)
        # y_sample = bytes_to_int(y_train[sorted_distance_indices[0]])
        y_pred.append(top_candidate)
    # print(y_pred)
    return y_pred


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, N_TRAIN_IMAGES)
    y_train = read_labels(TRAIN_LABELS_FILENAME, N_TRAIN_IMAGES)
    # [0,1,9,5,...]
    # X_test = read_images(TEST_DATA_FILENAME, N_TEST_IMAGES)
    # True solution, we won't have it for a real project, we use it to check the accuracy of the model
    y_test = read_labels(TEST_LABELS_FILENAME, N_TEST_IMAGES)



    if DEBUG:
        X_test = [read_image(f'{TEST_DIR}our_test.png')]
        y_test = [8]
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')


    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train,y_train, X_test,7)

    print(y_pred)

    correct_predictions = [
        y_pred_i == y_test_i
        for y_pred_i, y_test_i
        in zip(y_pred, y_test)
    ]
    print(correct_predictions)
    prediction_accuracy = sum(correct_predictions)/len(y_pred)
    print(prediction_accuracy)

if __name__ == '__main__':
    main()
