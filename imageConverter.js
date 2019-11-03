const fs = require('fs'),
  tf = require('@tensorflow/tfjs-node');

class ImageConverter {
  getDirFilesList(dirPath) {
    return fs.readdirSync(dirPath);
  }

  convertImageToUInt8Array(dirPath, fileName) {
    let image = fs.readFileSync(`${dirPath}${fileName}`);  
    return new Uint8Array(image);
  }

  decodeImgFromUInt8ArrayToTensor(array) {
    return tf.node.decodeImage(array, 3);
  }
}

module.exports = ImageConverter;