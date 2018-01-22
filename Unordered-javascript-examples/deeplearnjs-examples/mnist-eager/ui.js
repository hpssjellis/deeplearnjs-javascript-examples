

const mathUi = dl.ENV.math;

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

/*

function isTraining() {
  statusElement.innerText = 'Training...';
}


async function trainingLog(message) {
  messageElement.innerText = await message + "\n";
 console.log(await message);
}



*/


function showTestResults(batch, predictions, labels) {
    statusElement.innerText = 'Testing...';
    var testExamples = batch.xs.shape[0];
    var totalCorrect = 0;
    for (var i = 0; i < testExamples; i++) {
        var image = mathUi.slice2D(batch.xs, [i, 0], [1, batch.xs.shape[1]]);
        var div = document.createElement('div');
        div.className = 'pred-container';
        var canvas = document.createElement('canvas');
        draw(image.flatten(), canvas);
        var pred = document.createElement('div');
        var prediction = predictions[i];
        var label = labels[i];
        var correct = prediction === label;
        if (correct) {
            totalCorrect++;
        }
        pred.className = "pred " + (correct ? 'pred-correct' : 'pred-incorrect');
        pred.innerText = "pred: " + prediction;
        div.appendChild(pred);
        div.appendChild(canvas);
        imagesElement.appendChild(div);
    }
    var accuracy = 100 * totalCorrect / testExamples;
    var displayStr = "accuracy: " + accuracy.toFixed(2) + "% (" + totalCorrect + " / " + testExamples + ")";
    messageElement.innerText = displayStr + "\n";
    console.log(displayStr);
}

function draw(image, canvas) {
 //const [width, height] = [28, 28];
  const width= 28;
  const height = 28;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
