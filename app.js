const tf = require('@tensorflow/tfjs-node'),
  CNN = require('./cnn.js'),
  Data = require('./data.js');

const cnn = new CNN(),
  data = new Data();

async function run() {
  let model = cnn.createCNNModel();
  let trainData = data.loadData('./imgFolder/');
  let history = await cnn.trainModel(model, trainData.data, trainData.labels);
  let pred = prepareDataForPrediction('./imgFolder/', 'l2.jpg');
  model.predict(pred.reshape([1, 200, 200, 3]), {batchSize: 1}).print();
}  

run();