const tf = require('@tensorflow/tfjs-node'),
  CNN = require('./cnn.js'),
  Data = require('./data.js');

const cnn = new CNN();  

async function run() {
  const data = new Data();
  await data.load();
  const model = cnn.createCNNModel();
  await cnn.trainCNNModel(model, data);
  
  const [p, l] = cnn.makePrediction(model, data, 10); 
  console.log('Predicitions');
  p.print(); 
  console.log('Labels'); 
  l.print();
}  

run();