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

  async getImagesUInt8Array(dirPath, imageSize, imageNumber){
    let files = fs.readdirSync('./imgFolder');
    let imageDataset = new Uint8Array(imageSize * imageNumber);
    for(let i = 0; i < files.length; i++) {
      let imageNameArray = new Uint8Array(fs.readFileSync(`${dirPath}${files[i]}`));
      for(let j = i * imageSize; j < i * imageSize + imageSize; j++) {
        imageDataset[j] = imageNameArray[j - i * imageSize];
      }      
    }
    
    return imageDataset;
  }
}

module.exports = ImageConverter;