const tf = require('@tensorflow/tfjs-node');

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
  constructor(){
    this.model = tf.sequential();
  }

  createCNNModel() {
    this.model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGTH, IMAGE_CHANNELS],
      filters: NUMBER_OF_FILTERS,
      kernelSize: KERNEL_SIZE,
      padding: 'valid'
    }));

    this.model.add(tf.layers.reLU());

    this.model.add(tf.layers.conv2d({
      filters: NUMBER_OF_FILTERS,
      kernelSize: KERNEL_SIZE
    }));

    this.model.add(tf.layers.reLU());

    this.model.add(tf.layers.maxPooling2d({
      poolSize: [MAX_POOLING, MAX_POOLING]
    }));

    this.model.add(tf.layers.dropout({
      rate: DROPOUT_VALUE
    }));

    this.model.add(tf.layers.flatten());

    this.model.add(tf.layers.dense({
      units: DIMENSIONALITY
    }));

    this.model.add(tf.layers.reLU());

    this.model.add(tf.layers.dropout({
      rate: DROPOUT_VALUE
    }));

    this.model.add(tf.layers.dense({
      units: NUMBER_OF_CLASSES
    }));

    this.model.add(tf.layers.softmax());

    this.model.compile({
      optimizer: 'adadelta',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    this.model.summary();
    this.model.getConfig();

    return this.model;
  }

  async trainCNNModel(model, data) {
    const BATCH_CONTAINER_SIZE = 1;
    const TRAIN_DATA_SIZE = 12;
    const TEST_DATA_SIZE = 3;

    const [trainD, trainL] = tf.tidy(() => {
      const bat = data.getTrainBatchContainer(TRAIN_DATA_SIZE);
      return [bat.xSet.reshape([TRAIN_DATA_SIZE, 200, 200, 3]), bat.labelsSet];
    });

    const [testD, testL] = tf.tidy(() => {
      const bat = data.getTestBatchContainer(TEST_DATA_SIZE);
      return [bat.xSet.reshape([TEST_DATA_SIZE, 200, 200, 3]), bat.labelsSet];
    });

    return model.fit(trainD, trainL, {
      batchSize: BATCH_CONTAINER_SIZE,
      epochs: 5,
      shuffle: true
    });
  }

  makePrediction(model, data, dataSize) {
    const testData = data.getTestBatchContainer(dataSize);
    const testD = testData.xSet.reshape([dataSize, 200, 200, 3]);
    const labels = testData.labelsSet;
    const predictions = model.predict(testD);
    testD.dispose();
    return [predictions, labels];
  }
}

module.exports = CNN;