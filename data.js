const tf = require('@tensorflow/tfjs-node'),
  ImageConverter = require('./ImageConverter.js');

const imgConv = new ImageConverter();

class Data{
  loadData(path){
    let labelsArr = [],
      dataTensorArr = [],
      files = imgConv.getDirFilesList(path);

    tf.util.shuffle(files);
    for(let i = 0; i < files.length; i++){
      if(files[i].includes('k')){
        labelsArr.push([1, 0])
      } else if(files[i].includes('l')) {
        labelsArr.push([0, 1])
      }
    }
    for(let i=0; i<files.length; i++) {
      dataTensorArr.push(imgConv.decodeImgFromUInt8ArrayToTensor(imgConv.convertImageToUInt8Array(path, files[i])));
    }

    return {
      labels: tf.tensor(labelsArr),
      data: dataTensorArr
    }
  }

  prepareDataForPrediction(path, file){
    return imgConv.decodeImgFromUInt8ArrayToTensor(imgConv.convertImageToUInt8Array(path, file));
  }
}

module.exports = Data;