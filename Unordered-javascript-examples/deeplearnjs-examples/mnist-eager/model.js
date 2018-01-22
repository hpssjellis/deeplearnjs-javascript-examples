


// Hyperparameters.
const LEARNING_RATE = .05;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;
// Data constants.
const IMAGE_SIZE = 784;
const LABELS_SIZE = 10;


//const math = dl.ENV.math;


const optimizer = new dl.SGDOptimizer(LEARNING_RATE);



// Set up the model and loss function.
const weights = dl.variable(dl.Array2D.randNormal([IMAGE_SIZE, LABELS_SIZE], 0, 1 / Math.sqrt(IMAGE_SIZE), 'float32'));


const model = (xs) => {
    return math.matMul(xs, weights);
};



const loss = (labels, ys) => {
    return math.mean(math.softmaxCrossEntropyWithLogits(labels, ys));
};



// Train the model.
async function train(data) {
    const returnCost = true;
    for (let i = 0; i < TRAIN_STEPS; i++) {
        const cost = optimizer.minimize(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE);
            return loss(batch.labels, model(batch.xs));
        }, returnCost);
        //document.getElementById('message').innerHTML = `loss[${i}]: ${cost.dataSync()}`
        document.getElementById('message').innerHTML = 'loss[' + i + ']:' +  await cost.dataSync()
        await dl.util.nextFrame();
    }
}



// Tests the model on a set
async function test(data) { }



// Predict the digit number from a batch of input images.
function predict(x) {
    const pred = math.scope(() => {
        const axis = 1;
        return math.argMax(model(x), axis);
    });
    return Array.from(pred.dataSync());
}




// Given a logits or label vector, return the class indices.
function classesFromLabel(y) {
    const axis = 1;
    const pred = math.argMax(y, axis);
    return Array.from(pred.dataSync());
}





