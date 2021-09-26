import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


def create_network():
    inputs = keras.layers.Input((64, 84, 1))
    pad = keras.layers.ZeroPadding2D(((0, 0), (0, 96 - 84)))(inputs)

    # First extract feature map  1/2
    conv1 = keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(pad)
    lk1 = keras.layers.LeakyReLU()(conv1)
    conv2 = keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(lk1)
    pool1 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv2)
    lk2 = keras.layers.LeakyReLU()(pool1)
    bn1 = keras.layers.BatchNormalization()(lk2)

    # Second extract feature map  1/4
    conv3 = keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(bn1)
    lk3 = keras.layers.LeakyReLU()(conv3)
    conv4 = keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(lk3)
    pool2 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv4)
    lk4 = keras.layers.LeakyReLU()(pool2)
    bn2 = keras.layers.BatchNormalization()(lk4)

    # Third extract feature map  1/8
    conv5 = keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(bn2)
    lk5 = keras.layers.LeakyReLU()(conv5)
    conv6 = keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(lk5)
    pool3 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv6)
    lk6 = keras.layers.LeakyReLU()(pool3)
    bn3 = keras.layers.BatchNormalization()(lk6)

    # Fourth extract feature map  1/16
    conv7 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn3)
    lk7 = keras.layers.LeakyReLU()(conv7)
    conv8 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(lk7)
    pool4 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv8)
    lk8 = keras.layers.LeakyReLU()(pool4)
    bn4 = keras.layers.BatchNormalization()(lk8)

    # Fifth extract feature map  1/32
    conv9 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn4)
    lk9 = keras.layers.LeakyReLU()(conv9)
    conv10 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(lk9)
    pool5 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv10)
    lk10 = keras.layers.LeakyReLU()(pool5)
    bn5 = keras.layers.BatchNormalization()(lk10)

    # Intermediate transition
    conv11 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn5)
    lk11 = keras.layers.LeakyReLU()(conv11)
    bn6 = keras.layers.BatchNormalization()(lk11)

    conv12 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn6)
    lk12 = keras.layers.LeakyReLU()(conv12)
    bn7 = keras.layers.BatchNormalization()(lk12)

    # First Deconvolution and expansion    1/16
    d_conv1 = keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(bn7)
    merge1 = keras.layers.concatenate([bn4, d_conv1])
    lk13 = keras.layers.LeakyReLU()(merge1)
    bn8 = keras.layers.BatchNormalization()(lk13)

    # Second Deconvolution and expansion    1/8
    d_conv2 = keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(bn8)
    merge2 = keras.layers.concatenate([bn3, d_conv2])
    lk14 = keras.layers.LeakyReLU()(merge2)
    bn9 = keras.layers.BatchNormalization()(lk14)

    # Third Deconvolution and expansion    1/4
    d_conv3 = keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(bn9)
    merge3 = keras.layers.concatenate([bn2, d_conv3])
    lk15 = keras.layers.LeakyReLU()(merge3)
    bn10 = keras.layers.BatchNormalization()(lk15)

    # Fourth Deconvolution and expansion    1/2
    d_conv4 = keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(bn10)
    merge4 = keras.layers.concatenate([bn1, d_conv4])
    lk16 = keras.layers.LeakyReLU()(merge4)
    bn11 = keras.layers.BatchNormalization()(lk16)

    # Fifth Deconvolution and expansion    1/1
    d_conv4 = keras.layers.Conv2DTranspose(11, kernel_size=5, strides=2, padding='same')(bn11)
    lk17 = keras.layers.LeakyReLU()(d_conv4)
    bn12 = keras.layers.BatchNormalization()(lk17)

    # Final process and Crop
    d_conv5 = keras.layers.Conv2DTranspose(11, kernel_size=5, strides=1, padding='same')(bn12)
    crop = keras.layers.Cropping2D(((0, 0), (0, 96 - 84)))(d_conv5)
    outputs = keras.layers.Activation('softmax')(crop)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


# batch generator: reduce the consumption of computer memory
def generator(train_x, train_y, batch_size):

    while 1:
        row = np.random.randint(0, len(train_x), size=batch_size)
        x = train_x[row]
        y = train_y[row]
        yield x, y


# create model and train and save
def train_network(train_x, train_y, test_x, test_y, epoch, batch_size):
    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis]

    model = create_network()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit_generator(generator(train_x, train_y, batch_size), epochs=epoch,
                        steps_per_epoch=len(train_x) // batch_size)

    score = model.evaluate(test_x, test_y, verbose=0)
    print('first_model test accuracy:', score[1])

    model.save('first_model.h5')


# Load the partially trained model and continue training and save
def load_network_then_train(train_x, train_y, test_x, test_y, epoch, batch_size, input_name, output_name):
    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis]

    model = load_model(input_name)
    history = model.fit_generator(generator(train_x, train_y, batch_size),
                                  epochs=epoch, steps_per_epoch=len(train_x) // batch_size)

    score = model.evaluate(test_x, test_y, verbose=0)
    print(output_name, 'test accuracy:', score[1])

    model.save(output_name)
    show_plot(history)


# plot the loss and the accuracy
def show_plot(history):
    # list all data in history
    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('loss1.jpg')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig('accuracy1.jpg')
    plt.show()


# show the real_img and the  network_training img
def plot_result(test_x, input_name, index):

    model = load_model(input_name)
    test_x = test_x[:, :, :, np.newaxis]
    net_result = model.predict(test_x)

    real_img = test_x[index]
    cv2.namedWindow("Real_Image")
    cv2.imshow("Real_Image", real_img)
    cv2.waitKey(0)
    cv2.imwrite('/home/archer/CODE/PF/real6.png', real_img * 255)

    net_cuboid = net_result[index]
    size = net_cuboid.shape
    mask = np.zeros((size[0], size[1]))
    net_img = np.zeros((size[0], size[1], 3))

    for i in range((size[0])):
        for j in range((size[1])):
            index = np.argmax(net_cuboid[i, j, :])
            mask[i, j] = int(index)

    print('Number in this picture contain :')
    print(np.unique(mask))

    # 0 - purplish red
    # 1 - orange
    # 2 - green
    # 3 - pink
    # 4 - white
    # 5 - gray
    # 6 - yellow
    # 7 - violet
    # 8 - dark blue
    # 9 - black
    # 10 -light blue

    colour = np.array([[255, 0, 255], [0, 0, 255], [0, 255, 0], [255, 192, 203],
                       [225, 255, 255], [155, 155, 155], [0, 255, 255], [120, 0, 128],
                       [255, 0, 0], [0, 0, 0], [255, 255, 0]])

    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if mask[i, j] == k:
                    net_img[i, j, :] = colour[k]/255

    cv2.namedWindow("Net_Image1")
    cv2.imshow("Net_Image1", net_img)
    cv2.waitKey(0)
    cv2.imwrite('/home/archer/CODE/PF/network6.png', net_img*255)
