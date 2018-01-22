//import {MnistData} from './data';
//import * as model from './model';
//import * as ui from './ui';

//data_new


var data;
async function myLoad() {
  data = new MnistData();      //from data_new.js
 // console.log(data)
  await data.load();
}

async function myTrain() {
  
   console.log('8')
  //ui.isTraining();                          //from ui.js
  //trainingLog('fred')     //note: isTraining and Training log have async issues
  document.getElementById('status').innerText = 'Training....'
  console.log('9')
  //await train(data, trainingLog);    //from model.js
  await train(data);    //from model.js
  
   console.log('10')
}

async function myTest() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const predictions = predict(batch.xs);            // from model.js
  const labels = classesFromLabel(batch.labels);   // from model.js

  showTestResults(batch, predictions, labels);    //from ui.js
}

async function myMnist() {
  await myLoad();
  await myTrain();
  myTest();
}

myMnist();
