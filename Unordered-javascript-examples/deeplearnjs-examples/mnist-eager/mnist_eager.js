


let data;


async function myLoad() {
    document.getElementById('status').innerText += ' Loading...'
    data = new MnistData();
    await data.load();
}


async function myTrain() {
    document.getElementById('status').innerText += ' Training...'
    await train(data);
}



async function myTest() {
    //document.getElementById('status').innerText += ' Testing...'
    const testExamples = 50;
    const batch = data.nextTestBatch(testExamples);
    const predictions = predict(batch.xs);
    const labels = classesFromLabel(batch.labels);
    showTestResults(batch, predictions, labels);
}


async function myMnist() {
    await myLoad();
    await myTrain();
    myTest();
}



