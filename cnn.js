const tf = require('@tensorflow/tfjs-node'),
  ImageConverter = require('./imageConverter.js');

const imgConv = new ImageConverter(),  
  model = tf.sequential();

//Total number of convolutional filters to use
const NUMBER_OF_FILTERS = 32;
//Size of kernel matrix
const KERNEL_SIZE = 3;
//For color image 3 channels (RGB)
const IMAGE_CHANNELS = 3;
//Input images size
const IMAGE_WIDTH = 200;
const IMAGE_HEIGTH = 200;
const MAX_POOLING = 2;
const DROPOUT_VALUE = 0.5;
const DIMENSIONALITY = 128;
//Number of gestures
const NUMBER_OF_CLASSES = 2;

class CNN {
  createCNNModel() {
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGTH, IMAGE_CHANNELS],
      filters: NUMBER_OF_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: 'valid'
    }));

    model.add(tf.layers.reLU());

    model.add(tf.layers.conv2d({
      filters: NUMBER_OF_FILTERS,
      kernelSize: KERNEL_SIZE
    }));

    model.add(tf.layers.reLU());

    model.add(tf.layers.maxPooling2d({
      poolSize: [MAX_POOLING, MAX_POOLING]
    }));

    model.add(tf.layers.dropout({
      rate: DROPOUT_VALUE
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
      units: DIMENSIONALITY
    }));

    model.add(tf.layers.reLU());

    model.add(tf.layers.dropout({
      rate: DROPOUT_VALUE
    }));

    model.add(tf.layers.dense({
      units: NUMBER_OF_CLASSES
    }));

    model.add(tf.layers.softmax());

    model.compile({
      optimizer: 'adadelta',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    model.summary();
    model.getConfig();

    return model;
  }
}