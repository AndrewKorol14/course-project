const tf = require('@tensorflow/tfjs-node'),
  ImageConverter = require('./ImageConverter.js'),
  labelsJson = require('./labels.json');

const imgConv = new ImageConverter();

const COMMON_IMAGE_SIZE = 200 * 200 * 3;

const CLASS_QUANTITY = 2;

const ALL_ELEMENT_QUANTITY = 11;
const TRAIN_ELEMENT_QUANTITY = 7;
const TEST_ELEMENT_QUANTITY = 4;

class Data {
  constructor(){
    this.trainInd = [];
    this.testInd = [];

    this.shufTrainInd = 0;
    this.shufTestInd = 0;
  }

async load() {
    let labelsArr = new Uint8Array(labelsJson.labels),
      imagesArr = await imgConv.getImagesUInt8Array('./imgFolder/', COMMON_IMAGE_SIZE, ALL_ELEMENT_QUANTITY);
    
    this.trainInd = tf.util.createShuffledIndices(TRAIN_ELEMENT_QUANTITY);
    this.testInd = tf.util.createShuffledIndices(TEST_ELEMENT_QUANTITY);

    this.trainImages = imagesArr.slice(0, COMMON_IMAGE_SIZE * TRAIN_ELEMENT_QUANTITY);
    this.testImages = imagesArr.slice(COMMON_IMAGE_SIZE * TRAIN_ELEMENT_QUANTITY); //console.log(imagesArr.length);

    this.trainLabels = labelsArr.slice(0, CLASS_QUANTITY * TRAIN_ELEMENT_QUANTITY);
    this.testLabels = labelsArr.slice(CLASS_QUANTITY * TRAIN_ELEMENT_QUANTITY); //console.log(this.trainLabels);
  }

  getBatchContainer(size, data, indexCallback){
    let batchContainerImgArr = new Float32Array(size * COMMON_IMAGE_SIZE);
    let batchContainerLabArr = new Uint8Array(size * CLASS_QUANTITY);

    for(let i = 0; i < size; i++) {
      let index = indexCallback();

      let img = data[0].slice(index * COMMON_IMAGE_SIZE, index * COMMON_IMAGE_SIZE + COMMON_IMAGE_SIZE);
      batchContainerImgArr.set(img, i * COMMON_IMAGE_SIZE);

      let lab = data[1].slice(index * CLASS_QUANTITY, index * CLASS_QUANTITY + CLASS_QUANTITY);
      batchContainerLabArr.set(lab, i * CLASS_QUANTITY);
    }

    let xSet = tf.tensor2d(batchContainerImgArr, [size, COMMON_IMAGE_SIZE]);
    let labelsSet = tf.tensor2d(batchContainerLabArr, [size, CLASS_QUANTITY]);

    return {xSet, labelsSet}; 
  }

  getTrainBatchContainer(size) {
    return this.getBatchContainer(size, [this.trainImages, this.trainLabels], () => {
      this.shufTrainInd = (this.shufTrainInd + 1) % this.trainInd.length;
      return this.trainInd[this.shufTrainInd];
    });
  }

  getTestBatchContainer(size) {
    return this.getBatchContainer(size, [this.testImages, this.testLabels], () => {
      this.shufTestInd = (this.shufTestInd + 1) % this.testInd.length;
      return this.testInd[this.shufTestInd];
    });
  }
}

module.exports = Data;