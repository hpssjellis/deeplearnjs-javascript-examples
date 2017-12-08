var Graph = dl.Graph;
var Tensor = dl.Tensor;
var Scalar = dl.Scalar;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var Session = dl.Session;
var track = dl.track;
var keep = dl.keep;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var SGDOptimizer = dl.SGDOptimizer;
var MomentumOptimizer = dl.MomentumOptimizer;
var optimizer;
var learningRate;
var initialLearningRate;

var CostReduction = dl.CostReduction;
var Array1D = dl.Array1D;
var Array4D = dl.Array4D;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var util = dl.util;
var conv_util = dl.conv_util;
const DATASETS_CONFIG_JSON = 'deeplearnjs/mnist/model-builder-datasets-config.json';
var xhrDatasetConfigs;
var selectedDatasetName;

var VarianceScalingInitializer = dl.VarianceScalingInitializer;
var ZerosInitializer = dl.ZerosInitializer;

var inputShape;
var labelShape;
var chart;
var chartData = [];
var config;

var feedEntries;
var inferenceFeedEntries;

var costTensor;
var predictionTensor;
var accuracyTensor;

var math;

var IMAGE_DATA_INDEX = 0;
var LABEL_DATA_INDEX = 1;

const INFERENCE_EXAMPLE_COUNT = 1;
const INFERENCE_IMAGE_SIZE_PX = 100;

var btn;
var accuracyElt;
var mathCPU;
var mathGPU;
var request = false;
var paused = true;
var step = 0;
var datasetDownloaded = false;


function copyArrayToCanvas(ndarray, canvasId){

var ctx = document.getElementById(canvasId).getContext("2d");

// overrides whatever the canvas was on the html page
ctx.canvas.width = ndarray.shape[0]   
ctx.canvas.height = ndarray.shape[1]

var h = ctx.canvas.height;  
var w = ctx.canvas.width;

var imgData = ctx.getImageData(0, 0, w, h);
var data = imgData.data;
     
      // document.getElementById('myDivOut03').innerHTML = 'height = '+h
        for (let i = 0; i < ndarray.shape[0]; i++) {
            for (let j = 0; j < ndarray.shape[1]; j++) {
                var s = 4 * i * h + 4 * j;  // for correct location
                data[s]   =  ndarray.get(i, j, 0);
                data[s+1] =  ndarray.get(i, j, 0);  // 1 for RGB
                data[s+2] =  ndarray.get(i, j, 0);  // 2 for RGB
                data[s+3] = 255;
            }
        }

ctx.putImageData(imgData, 0, 0);

}





function createFullyConnectedLayer(
    graph, inputLayer, layerIndex,
    outputDepth, activation = (x) => graph.relu(x), useBias = true) {

    weightsInitializer = new VarianceScalingInitializer();
    biasInitializer = new ZerosInitializer();

    let out = graph.layers.dense(
        'fully_connected_' + layerIndex,
        inputLayer,
        outputDepth,
        activation,
        useBias,
        weightsInitializer,
        biasInitializer);

    return out;
}

function createConv2dLayer(graph, inputLayer, layerIndex, outputDepth, inputShape, filterSize = 2, stride = 2, zeroPad = 0, activation = (x) => graph.relu(x)) {

    // input must be Array3D: x,y,h

    const wShape = [filterSize, filterSize, inputShape[2], outputDepth];
    w = Array4D.randTruncatedNormal(wShape, 0, 0.1);
    b = Array1D.zeros([outputDepth]);

    const wTensor = graph.variable(`conv2d-${layerIndex}-w`, w);
    const bTensor = graph.variable(`conv2d-${layerIndex}-b`, b);

    let out = graph.conv2d(
        inputLayer, wTensor, bTensor, filterSize, outputDepth,
        stride, zeroPad);

    if (activation != null) {
        out = activation(out);
    }

    out.outputShape = conv_util.computeOutputShape3D(
        inputShape, filterSize, outputDepth, stride, zeroPad);

    return out;
}

function createMaxPoolLayer(graph, inputLayer, layerIndex, inputShape, filterSize = 2, stride = 2, zeroPad = 0) {
    let out = graph.maxPool(inputLayer, filterSize, stride, zeroPad);

    out.outputShape = conv_util.computeOutputShape3D(
        inputShape, filterSize, inputShape[2], stride, zeroPad);
    return out;
}

function createFlattenLayer(graph, inputLayer, layerIndex, inputShape) {

    let size = util.sizeFromShape(inputShape);
    let out = graph.reshape(inputLayer, [size]);

    out.outputShape = [size];
    return out;
}


function train1Batch(shouldFetchCost) {
    // Every 42 steps, lower the learning rate by 15%.
    learningRate = initialLearningRate * Math.pow(0.95, Math.floor(step / 82));
    optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    let metric = Scalar.new(0);
    math.scope(() => {
        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer,
            shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

        if (!shouldFetchCost) {
            // We only train. We do not compute the cost.
            return;
        }

        // Compute the cost (by calling get), which requires transferring data
        // from the GPU.
        costValue = cost.get();

        // calculate accuracy

        for (let i = 0; i < batchSize; i++) {
            const metricValue = session.eval(accuracyTensor, inferenceFeedEntries);

            metric = math.add(metric, metricValue);
        }

        metric = math.divide(metric, Scalar.new(batchSize));

        metric = metric.getValues();

    });

    return [costValue, metric];
}

function predict(predictionTensor, feedEntries, callback) {

    math.scope((keep, track) => {

        const feeds = [];
        const inferenceValues = [];

        const ndarrayFeedEntries = [];
        ndarrayFeedEntries.push({
            tensor: feedEntries[IMAGE_DATA_INDEX].tensor,
            data: track((feedEntries[IMAGE_DATA_INDEX].data).getNextCopy(math))
        }, {
            tensor: feedEntries[LABEL_DATA_INDEX].tensor,
            data: track((feedEntries[LABEL_DATA_INDEX].data).getNextCopy(math))
        })

        feeds.push(ndarrayFeedEntries);

        inferenceValues.push(session.eval(predictionTensor, ndarrayFeedEntries));

        inferenceValues[inferenceValues.length - 1].getValues();

        // values = Array.prototype.slice.call(values);

        // console.log('values2', values)

        callback(feeds, inferenceValues)

    });
    return
}


function displayInferenceExamplesOutput(inputFeeds, inferenceOutputs) {
    let images = [];
    const logits = [];
    const labels = [];

    for (let i = 0; i < inputFeeds.length; i++) {
        images.push(inputFeeds[i][IMAGE_DATA_INDEX].data);
        labels.push(inputFeeds[i][LABEL_DATA_INDEX].data);
        logits.push(inferenceOutputs[i]);
    }

    images = dataSet.unnormalizeExamples(images, IMAGE_DATA_INDEX);

    // Draw the images.
  //  for (let i = 0; i < inputFeeds.length; i++) {
   //     inputNDArrayVisualizers[i].saveImageDataFromNDArray(images[i]);
   // }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
        const softmaxLogits = math.softmax(logits[i]);

      // document.getElementById('myDivOut03').innerHTML ='Input length = ' + inputFeeds.length + 'Logits = '+ logits[i].getValues() + ', soft = ' + softmaxLogits.getValues() + '<br>'






     //   outputNDArrayVisualizers[i].drawLogits(
      //      softmaxLogits, labels[i],
      //      xhrDatasetConfigs[selectedDatasetName].labelClassNames);
     //   inputNDArrayVisualizers[i].draw();

        
        
    
        // Print data to console so the user can inspect.
       // document.getElementById('egdiv').innerHTML =  
   
       // Plots the image


        copyArrayToCanvas(images[i], 'myPlot')
        //document.getElementById('myDivOut03').innerHTML = 'length ' +images[i].length+ ' shape0 = '+images[i].shape[0] + 'shape1 = '+images[i].shape[1]  +'.......'+ images[i].getValues()
        //console.log('length ' +images[i].length+ ' shape0 = '+images[i].shape[0] + 'shape1 = '+images[i].shape[1]  +'.......'+ images[i].getValues())
       
       
        const mathCpu2 = new NDArrayMathCPU();
        const labelClass2 = mathCpu2.argMax(labels[i]).get();

        const topk2 = mathCpu2.topK(softmaxLogits, 10); //get 3 labels
        const topkIndices2 = topk2.indices.getValues()
        const topkValues2 = topk2.values.getValues()
        myOutString = 'Wrong'
        myOutColor = 'red'
        if(topkIndices2[0] == labelClass2){
            myOutString = 'Correct!'
            myOutColor = 'green'
        }
        document.getElementById('myDivOut04').innerHTML = 'Actual Label is ' + labelClass2 + '. Your program was <Font color=\''+myOutColor+'\'>'+myOutString + '</font><br>'+
                                                          topkIndices2[0] + ' = ' + Math.floor(topkValues2[0]*1000)/10 + '% <br>'+ 
                                                          topkIndices2[1] + ' = ' + Math.floor(topkValues2[1]*1000)/10+ '% <br>'+ 
                                                          topkIndices2[2] + ' = ' + Math.floor(topkValues2[2]*1000)/10 + '% <br>'+   
                                                          topkIndices2[3] + ' = ' + Math.floor(topkValues2[3]*1000)/10 + '% <br>'+   
                                                          topkIndices2[4] + ' = ' + Math.floor(topkValues2[4]*1000)/10 + '% <br>'+   
                                                          topkIndices2[5] + ' = ' + Math.floor(topkValues2[5]*1000)/10 + '% <br>'+   
                                                          topkIndices2[6] + ' = ' + Math.floor(topkValues2[6]*1000)/10 + '% <br>'+   
                                                          topkIndices2[7] + ' = ' + Math.floor(topkValues2[7]*1000)/10 + '% <br>'+   
                                                          topkIndices2[8] + ' = ' + Math.floor(topkValues2[8]*1000)/10 + '% <br>'+   
                                                          topkIndices2[9] + ' = ' + Math.floor(topkValues2[9]*1000)/10 + '% <br>'    
        
        
        softmaxLogits.dispose();
  }


}




/*

function createChart(canvasElt, label, data, min = 0, max = null) {

    config = {
        type: 'line',
        data: {
            datasets: [{
                data: [],
                fill: false,
                label: ' ',
                pointRadius: 0,
                borderColor: 'rgba(75,192,192,1)',
                backgroundColor: 'rgba(75,192,192,1)',
                borderWidth: 1,
                // lineTension: 0,
                // pointHitRadius: 8
            }]
        },
        options: {
            animation: {
                duration: 0
            },
            responsive: true,
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [{
                    ticks: {
                        min: null,
                        callback: (label, index, labels) => {
                            let num = Number(label).toFixed(2);
                            return `${num}`;
                        }
                    }
                }]
            }
        }
    };

    const context = canvasElt.getContext('2d');

    config.data.datasets[0].data = data;
    config.data.datasets[0].label = label;

    return new Chart(context, config);
}


*/




function buildModel() {


//*********************






  //  const inferenceContainer =
   //     document.querySelector('#inference-container');
  //  inferenceContainer.innerHTML = '';
 //   inputNDArrayVisualizers = [];
  //  outputNDArrayVisualizers = [];
  //  for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
  //      const inferenceExampleElement = document.createElement('div');
  //      inferenceExampleElement.className = 'inference-example';




        // Set up the input visualizer.

      //  const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement);
      //  ndarrayImageVisualizer.setShape(inputShape);
      //  ndarrayImageVisualizer.setSize(
      //      INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);

     //   inputNDArrayVisualizers.push(ndarrayImageVisualizer);
        // inferenceExampleElement.appendChild(ndarrayImageVisualizer);





        // Set up the output ndarray visualizer.
      //  const ndarrayLogitsVisualizer = new NDArrayLogitsVisualizer(inferenceExampleElement, 3);
     //   document.createElement('ndarray-logits-visualizer');
      //  ndarrayLogitsVisualizer.initialize(
      //      INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      //  outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
        // inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

     //   inferenceContainer.appendChild(inferenceExampleElement);
  //  }




//************************


    math = mathGPU;

    graph = new Graph();

    session = new Session(graph, math);

    // This tensor contains the input. In this case, it is a scalar.
    inputTensor = graph.placeholder('input', inputShape);

    // This tensor contains the target.
    targetTensor = graph.placeholder('label', labelShape);

    const [inputArray, targetArray] = dataSet.getData();

    const TRAIN_TEST_RATIO = 5 / 6;

    const threshold = Math.floor(TRAIN_TEST_RATIO * inputArray.length);

    shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([inputArray.slice(0, threshold), targetArray.slice(0, threshold)]);

    [inputProvider, targetProvider] = shuffledInputProviderBuilder.getInputProviders();

    feedEntries = [{
            tensor: inputTensor,
            data: inputProvider
        },
        {
            tensor: targetTensor,
            data: targetProvider
        }
    ];


    const inferenceShuffledInputProviderGenerator =
        new InCPUMemoryShuffledInputProviderBuilder([inputArray.slice(threshold), targetArray.slice(threshold)]);
    const [inferenceInputProvider, inferenceLabelProvider] =
    inferenceShuffledInputProviderGenerator.getInputProviders();

    inferenceFeedEntries = [{
            tensor: inputTensor,
            data: inferenceInputProvider
        },
        {
            tensor: targetTensor,
            data: inferenceLabelProvider
        }
    ];

    let net = inputTensor;
    let layerIndex = 0;

   // let netType = 'Fully Connected';
    let netType = 'Convolutional';

    if (netType == 'Fully Connected') {

        net = createFlattenLayer(graph, net, layerIndex++, inputShape);
        net = createFullyConnectedLayer(graph, net, layerIndex++, 128); // hidden layer

    } else if (netType == 'Convolutional') {

        net = createConv2dLayer(graph, net, layerIndex++, 16, inputShape, 5, 1, 2); // hidden layer
        net = createMaxPoolLayer(graph, net, layerIndex++, net.outputShape, 2, 2, 0);
        net = createFlattenLayer(graph, net, layerIndex++, net.outputShape);

    }

    predictionTensor = createFullyConnectedLayer(graph, net, layerIndex++, labelShape[0]);
    costTensor = graph.softmaxCrossEntropyCost(predictionTensor, targetTensor);
    accuracyTensor = graph.argmaxEquals(predictionTensor, targetTensor);



    batchSize = 30;
    initialLearningRate = 0.1;
    // optimizer = new SGDOptimizer(initialLearningRate);
    var momentum = 0.1;
    optimizer = new MomentumOptimizer(initialLearningRate, momentum);

    modelInitialized = true;
}

function populateDatasets(callback) {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(_xhrDatasetConfigs => {
                for (const datasetName in _xhrDatasetConfigs) {
                    if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                        dataSets[datasetName] =
                            new XhrDataset(_xhrDatasetConfigs[datasetName]);
                    }
                }
                var datasetNames = Object.keys(dataSets);
                xhrDatasetConfigs = _xhrDatasetConfigs;
                selectedDatasetName = datasetNames[0]
                dataSet = dataSets[selectedDatasetName]; //MNIST

                dataSet.fetchData().then(() => {
                    datasetDownloaded = true;
                    dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1)
                    callback();
                });

                inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
                labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);
               
            },
            error => {
                throw new Error(`Dataset config could not be loaded: ${error}`);
            });
}



function run() {

    mathGPU = new NDArrayMathGPU();
  //  mathCPU = new NDArrayMathCPU();

  //  chartData = []

  //  const canvasElt = document.getElementById('plot');
   // chart = createChart(canvasElt, 'cost', chartData, 0, chartData.y);
   // chart.update();

    btn.addEventListener('click', () => {
        // Activate, deactivate hyper parameter inputs.
        paused = !paused;
        if (paused == false) {
            request = true;
        }
       //  ga('send', 'event', 'deeplearn_mnist_starter', 'click', 'Start', 50);
    });



    // DOM setup
   // learningRateBtn = document.getElementById("learningRateBtn");
   // learningRateBtn.addEventListener('click', () => {
        // Activate, deactivate hyper parameter inputs.
        initialLearningRate = 0.1     //parseFloat(document.getElementById("learning-rate-input").value);
   // });




        math = 'mathGPU'

       // console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU')
       accuracyElt.innerHTML += 'math ='+ math
/*

    function updateSelectedEnvironment(selectedEnvName) {
        math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
        console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU')
    }
    var envDropdown = document.getElementById("environment-dropdown");
    var selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName);

    envDropdown.addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName);
    });
    
    
    */
    
    

    populateDatasets(buildModel);
}


function train_per() {
    if (step > 4242) {
        // Stop training.
        return;
    }

    if (paused) return;

    // We only fetch the cost every 10 steps because doing so requires a transfer
    // of data from the GPU.

    const [cost, accuracy] = train1Batch(step % 10 === 0);

    //var d = document.getElementById('egdiv');
   // d.innerHTML = 'step = ' + step;

    if (step % 20 === 0) {

        chartData.push({
            x: step,
            y: cost
        });

       // config.data.datasets[0].data = chartData;
       // chart.update();

        // Print data to console so the user can inspect.
        document.getElementById('egdiv').innerHTML ='Steps = '+ step + ', cost = '+ cost.toFixed(3) + ', accuracy = '+ accuracy[0].toFixed(3) +', Learning Rate = '+learningRate.toFixed(3) 

        accuracyElt.innerHTML = 'Accuracy = '+ Math.floor(accuracy[0]*100) + '%';

       // predict(predictionTensor, inferenceFeedEntries, displayInferenceExamplesOutput);

    }

    step++;

    requestAnimationFrame(() => train_per());
}


var modelInitialized = false;

function monitor() {

    if (datasetDownloaded == false) {
        btn.disabled = true;
        btn.value = 'Downloading data ...';

    } else {

        if (modelInitialized) {

            btn.disabled = false;

            if (paused) {
                btn.value = 'Start'
            } else {
                btn.value = 'Stop'
            }

            if (request) {
                request = false;
                train_per();
            }

         //   document.getElementById('learning-rate-input').value = learningRate;
           // document.getElementById('egdiv').innerHTML = 'step = ' + step;

        } else {
            btn.disabled = true;
            btn.value = 'Initializing Model ...'
        }
    }

    setTimeout(function () {
        monitor();
    }, 100);
}

function start() {

    supported = true //detect_support();

    btn = document.getElementById("buttontp");

    if (supported) {
       // console.log('device & webgl supported')
        accuracyElt = document.getElementById('accuracy');
       accuracyElt.innerHTML = 'device & webgl supported,   '
        btn.disabled = false;

        run();
        monitor();

    } else {
        console.log('device/webgl not supported')
        btn.disabled = true;
    }

}
