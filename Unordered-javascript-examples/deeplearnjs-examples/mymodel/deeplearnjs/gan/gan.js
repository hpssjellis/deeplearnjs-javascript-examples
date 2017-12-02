/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */



function insertLayerTableRow(elt, name, inShape, outShape) {

    elt.classList.add('table-responsive');
    elt.style.border = 'none';

    var h = document.createElement("h5");
    h.appendChild(document.createTextNode(`${name}`));

    var table = document.createElement('table');
    // table.classList.add('table');
    table.style.width = "100px";

    var head = table.createTHead();
    var row = head.insertRow(0);

    row.insertCell(0).outerHTML = `<th>in</th>`;
    row.insertCell(1).outerHTML = `<th>out</th>`;

    var body = table.createTBody();
    var row = body.insertRow(0);

    row.insertCell(0).innerHTML = `${inShape}`;
    row.insertCell(1).innerHTML = `${outShape}`;

    elt.appendChild(h);
    elt.appendChild(table);
}

function getRandomInputProvider(shape) {
    return {
        getNextCopy(math) {
            return NDArray.randNormal(shape);
        },
        disposeCopy(math, copy) {
            copy.dispose();
        }
    }
}

function getDisplayShape(shape) {
    return `[${shape}]`;
}

var Normalization = {
    NORMALIZATION_NEGATIVE_ONE_TO_ONE: 0,
    NORMALIZATION_ZERO_TO_ONE: 1,
    NORMALIZATION_NONE: 2
}

var dl = deeplearn;
var Array1D = dl.Array1D;
var Array3D = dl.Array3D;
var DataStats = dl.DataStats;
var FeedEntry = dl.FeedEntry;
var Graph = dl.Graph;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var Initializer = dl.Initializer;
var InMemoryDataset = dl.InMemoryDataset;
var MetricReduction = dl.MetricReduction;
var MomentumOptimizer = dl.MomentumOptimizer;
var SGDOptimizer = dl.SGDOptimizer;
var RMSPropOptimizer = dl.RMSPropOptimizer;
var AdagradOptimizer = dl.AdagradOptimizer;
var AdadeltaOptimizer = dl.AdadeltaOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
var AdamaxOptimizer = dl.AdamaxOptimizer;
var NDArray = dl.NDArray;
var NDArrayMath = dl.NDArrayMath;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Optimizer = dl.Optimizer;
var OnesInitializer = dl.OnesInitializer;
var Scalar = dl.Scalar;
var Session = dl.Session;
var Tensor = dl.Tensor;
var util = dl.util;
var VarianceScalingInitializer = dl.VarianceScalingInitializer;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var XhrDatasetConfig = dl.XhrDatasetConfig;
var ZerosInitializer = dl.ZerosInitializer;


// import {
//     NDArrayImageVisualizer
// } from './ndarray-image-visualizer';
// import {
//     NDArrayLogitsVisualizer
// } from './ndarray-logits-visualizer';
// import {
//     PolymerElement,
//     PolymerHTMLElement
// } from './polymer-spec';

// import {
//     LayerBuilder,
//     LayerWeightsDict
// } from './layer_builder';
// import {
//     ModelLayer
// } from './model-layer';
// import * as model_builder_util from './model_builder_util';
// import {
//     Normalization
// } from './tensorflow';
// import {
//     getRandomInputProvider
// } from './my_input_provider';
// import {
//     MyGraphRunner,
//     MyGraphRunnerEventObserver
// } from './my_graph_runner';

const DATASETS_CONFIG_JSON = 'deeplearnjs/gan/model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 1;
const INFERENCE_IMAGE_SIZE_PX = 100;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
};

// export class GANPlayground extends GANPlaygroundPolymer {
// Polymer properties.
var isValid;
var totalTimeSec;
var applicationState;
var modelInitialized;
var showTrainStats;
var selectedNormalizationOption = Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE;

// Datasets and models.
var graphRunner;
var graph;
var session;
var discOptimizer;
var genOptimizer;
var xTensor;
var labelTensor;
var costTensor;
var accuracyTensor;
var predictionTensor;
var discPredictionReal;
var discPredictionFake;
var discLoss;
var genLoss;
var generatedImage;

var datasetDownloaded;
var datasetNames;
var selectedEnvName;
var selectedDatasetName;
var modelNames;
var genModelNames;
var selectedModelName;
var genSelectedModelName;
var optimizerNames;
var discSelectedOptimizerName;
var genSelectedOptimizerName;
var loadedWeights;
var dataSets;
var dataSet;
var xhrDatasetConfigs;
var datasetStats;
var discLearningRate;
var genLearningRate;
var discMomentum;
var genMomentum;
var discNeedMomentum;
var genNeedMomentum;
var discGamma;
var genGamma;
var discBeta1;
var discBeta2;
var genBeta1;
var genBeta2;
var discNeedGamma;
var genNeedGamma;
var discNeedBeta;
var genNeedBeta;
var batchSize;

// Stats.
var showDatasetStats;
var statsInputRange;
var statsInputShapeDisplay;
var statsLabelShapeDisplay;
var statsExampleCount;

// Charts.
var costChart;
var accuracyChart;
var examplesPerSecChart;
var costChartData;
var accuracyChartData;
var examplesPerSecChartData;

var trainButton;

// Visualizers.
var inputNDArrayVisualizers;
var outputNDArrayVisualizers;

var inputShape;
var labelShape;
var randVectorShape;
var examplesPerSec;
var examplesTrained;
var inferencesPerSec;
var inferenceDuration;
var generationsPerSec;
var generationDuration;

var inputLayer;
var hiddenLayers;

var layersContainer;
var discHiddenLayers;
var genHiddenLayers;

var math;
// Keep one instance of each NDArrayMath so we don't create a user-initiated
// number of NDArrayMathGPU's.
var mathGPU = new NDArrayMathGPU();;
var mathCPU = new NDArrayMathCPU();;

function isTraining(applicationState) {
    return applicationState === ApplicationState.TRAINING;
}

function isIdle(applicationState) {
    return applicationState === ApplicationState.IDLE;
}



function getTestData() {
    const data = dataSet.getData();
    if (data == null) {
        return null;
    }
    const [images, labels] = dataSet.getData();

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
}

function getTrainingData() {
    const [images, labels] = dataSet.getData();

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
}

function getData() {
    return dataSet.getData();
}

function getImageDataOnly() {
    const [images, labels] = dataSet.getData();
    return images
}

function startInference() {
    const data = getImageDataOnly();
    if (data == null) {
        return;
    }
    if (isValid && (data != null)) {
        const shuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder([data]);
        const [inputImageProvider] =
        shuffledInputProviderGenerator.getInputProviders();

        const oneInputProvider = {
            getNextCopy(math) {
                return Array1D.new([0, 1]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const zeroInputProvider = {
            getNextCopy(math) {
                return Array1D.new([1, 0]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const inferenceFeeds = [{
                tensor: xTensor,
                data: inputImageProvider
            },
            {
                tensor: randomTensor,
                data: getRandomInputProvider(randVectorShape)
            },
            {
                tensor: oneTensor,
                data: oneInputProvider
            },
            {
                tensor: zeroTensor,
                data: zeroInputProvider
            }
        ]

        graphRunner.infer(
            generatedImage, discPredictionFake, discPredictionReal,
            inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS, INFERENCE_EXAMPLE_COUNT
        );
    }
}

function resetHyperParamRequirements(which) {
    if (which === 'gen') {
        genNeedMomentum = false;
        genNeedGamma = false;
        genNeedBeta = false;
    } else {
        discNeedMomentum = false;
        discNeedGamma = false;
        discNeedBeta = false;
    }
}

/**
 * Set flag to disable input by optimizer selection.
 */
function refreshHyperParamRequirements(optimizerName,
    which) {
    resetHyperParamRequirements(which);
    switch (optimizerName) {
        case "sgd":
            {
                // No additional hyper parameters
                break;
            }
        case "momentum":
            {
                if (which === 'gen') {
                    genNeedMomentum = true;
                } else {
                    discNeedMomentum = true;
                }
                break;
            }
        case "rmsprop":
            {
                if (which === 'gen') {
                    genNeedMomentum = true;
                    genNeedGamma = true;
                } else {
                    discNeedMomentum = true;
                    discNeedGamma = true;
                }
                break;
            }
        case "adagrad":
            {
                break;
            }
        case 'adadelta':
            {
                if (which === 'gen') {
                    genNeedGamma = true;
                } else {
                    discNeedGamma = true;
                }
                break;
            }
        case 'adam':
            {
                if (which === 'gen') {
                    genNeedBeta = true;
                } else {
                    discNeedBeta = true;
                }
                break;
            }
        case 'adamax':
            {
                if (which === 'gen') {
                    genNeedBeta = true;
                } else {
                    discNeedBeta = true;
                }
                break;
            }
        default:
            {
                throw new Error(`Unknown optimizer`);
            }
    }
}

function createOptimizer(which) {
    if (which === 'gen') {
        var selectedOptimizerName = genSelectedOptimizerName;
        var learningRate = genLearningRate;
        var momentum = genMomentum;
        var gamma = genGamma;
        var beta1 = genBeta1;
        var beta2 = genBeta2;
        var varName = 'generator';
    } else {
        var selectedOptimizerName = discSelectedOptimizerName;
        var learningRate = discLearningRate;
        var momentum = discMomentum;
        var gamma = discGamma;
        var beta1 = discBeta1;
        var beta2 = discBeta2;
        var varName = 'discriminator';
    }
    switch (selectedOptimizerName) {
        case 'sgd':
            {
                return new SGDOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'momentum':
            {
                return new MomentumOptimizer(+learningRate, +momentum,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'rmsprop':
            {
                return new RMSPropOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adagrad':
            {
                return new AdagradOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adadelta':
            {
                return new AdadeltaOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adam':
            {
                return new AdamOptimizer(+learningRate, +beta1, +beta2,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        default:
            {
                throw new Error(`Unknown optimizer`);
            }
    }
}

function startTraining() {
    const data = getImageDataOnly();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    discOptimizer = createOptimizer('disc');
    genOptimizer = createOptimizer('gen');

    if (isValid && data != null) {
        // recreateCharts();
        graphRunner.resetStatistics();

        const shuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder([data]);
        const [inputImageProvider] =
        shuffledInputProviderGenerator.getInputProviders();

        const oneInputProvider = {
            getNextCopy(math) {
                return Array1D.new([0, 1]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const zeroInputProvider = {
            getNextCopy(math) {
                return Array1D.new([1, 0]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const discFeeds = [{
                tensor: xTensor,
                data: inputImageProvider
            },
            {
                tensor: randomTensor,
                data: getRandomInputProvider(randVectorShape)
            },
            {
                tensor: oneTensor,
                data: oneInputProvider
            },
            {
                tensor: zeroTensor,
                data: zeroInputProvider
            }
        ]

        const genFeeds = [{
                tensor: randomTensor,
                data: getRandomInputProvider(randVectorShape)
            },
            {
                tensor: oneTensor,
                data: oneInputProvider
            },
            {
                tensor: zeroTensor,
                data: zeroInputProvider
            }
        ]

        graphRunner.train(
            discLoss, genLoss, discFeeds, genFeeds, batchSize,
            discOptimizer, genOptimizer, undefined, COST_INTERVAL_MS);

        showTrainStats = true;
        applicationState = ApplicationState.TRAINING;
    }
}

function createModel() {
    if (session != null) {
        session.dispose();
    }

    modelInitialized = false;
    if (isValid === false) {
        return;
    }

    // Construct graph
    graph = new Graph();
    const g = graph;
    randomTensor = g.placeholder('random', randVectorShape);
    xTensor = g.placeholder('input', inputShape);
    oneTensor = g.placeholder('one', [2]);
    zeroTensor = g.placeholder('zero', [2]);

    const varianceInitializer = new VarianceScalingInitializer()
    const zerosInitializer = new ZerosInitializer()
    const onesInitializer = new OnesInitializer();


    let weights = null;
    if (loadedWeights != null) {

        function toArray(dicValues, dicSize) {
            var array = dicValues;
            array.length = dicSize;
            return Array.prototype.slice.call(array);
        }

        console.log('loading weights', loadedWeights);
        weights = {};
        for (var key in loadedWeights) {
            weights[key] = toArray(loadedWeights[key].ndarrayData.values, loadedWeights[key].size);
        }

    } else {
        console.log('no weights loaded, random initialize weights');
    }

    // Construct generator
    let gen = randomTensor;
    for (let i = 0; i < genHiddenLayers.length; i++) {

        [gen] = genHiddenLayers[i].addLayerMultiple(g, [gen],
            'generator-' + i.toString(), weights);
    }
    gen = g.tanh(gen);

    // Construct discriminator
    let disc1 = gen;
    let disc2 = xTensor;
    for (let i = 0; i < discHiddenLayers.length; i++) {
        let weights = null;
        // if (loadedWeights != null) {
        //     weights = loadedWeights;
        // }
        [disc1, disc2] = discHiddenLayers[i].addLayerMultiple(g, [disc1, disc2],
            'discriminator' + i.toString(), weights);
    }

    discPredictionReal = disc2;
    discPredictionFake = disc1;
    generatedImage = gen;
    const discLossReal = g.softmaxCrossEntropyCost(
        discPredictionReal,
        oneTensor
    );
    const discLossFake = g.softmaxCrossEntropyCost(
        discPredictionFake,
        zeroTensor
    );
    discLoss = g.add(discLossReal, discLossFake);

    genLoss = g.softmaxCrossEntropyCost(
        discPredictionFake,
        oneTensor
    );

    session = new Session(g, math);
    graphRunner.setSession(session);

    // startInference();

    modelInitialized = true;
}

function populateDatasets() {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(
            _xhrDatasetConfigs => {
                for (const datasetName in _xhrDatasetConfigs) {
                    if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                        dataSets[datasetName] =
                            new XhrDataset(_xhrDatasetConfigs[datasetName]);
                    }
                }
                datasetNames = Object.keys(dataSets);
                selectedDatasetName = datasetNames[0]; // initialize on MNIST
                xhrDatasetConfigs = _xhrDatasetConfigs;
                updateSelectedDataset(selectedDatasetName);
            },
            error => {
                throw new Error('Dataset config could not be loaded: ' + error);
            });
}

function updateSelectedDataset(datasetName) {
    if (dataSet != null) {
        dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    graphRunner.stopTraining();
    graphRunner.stopInferring();

    if (dataSet != null) {
        dataSet.dispose();
    }

    selectedDatasetName = datasetName;
    selectedModelName = '';
    dataSet = dataSets[datasetName];
    datasetDownloaded = false;
    showDatasetStats = false;

    dataSet.fetchData().then(() => {
        datasetDownloaded = true;
        applyNormalization(selectedNormalizationOption);
        setupDatasetStats();
        if (isValid) {
            createModel();
        }
        // Get prebuilt models.
        populateModelDropdown();
    });

    inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
    //labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);
    labelShape = [2];

    layersContainer =
        document.querySelector('#hidden-layers');
    genLayersContainer =
        document.querySelector('#gen-hidden-layers');

    // DISC
    inputLayer = document.querySelector('#input-layer');
    insertLayerTableRow(inputLayer, 'input-layer', null, getDisplayShape(inputShape));

    const labelShapeDisplay =
        getDisplayShape(labelShape);
    const costLayer = document.querySelector('#cost-layer');
    insertLayerTableRow(costLayer, 'cost-layer', labelShapeDisplay, labelShapeDisplay);

    const outputLayer = document.querySelector('#output-layer');
    insertLayerTableRow(outputLayer, 'output-layer', labelShapeDisplay, null);

    // GEN
    genInputLayer = document.querySelector('#gen-input-layer');
    // genInputLayer.outputShapeDisplay =
    //     getDisplayShape(randVectorShape);
    insertLayerTableRow(genInputLayer, 'gen-input-layer', null, getDisplayShape(randVectorShape));

    const genCostLayer = document.querySelector('#gen-cost-layer');
    insertLayerTableRow(genCostLayer, 'gen-cost-layer', getDisplayShape(inputShape), getDisplayShape(inputShape));

    const genOutputLayer = document.querySelector('#gen-output-layer');
    insertLayerTableRow(genOutputLayer, 'gen-output-layer', labelShapeDisplay, null);

    buildRealImageContainer();
    buildFakeImageContainer();
}

/* Helper function for building out container for images*/
function buildRealImageContainer() {
    const inferenceContainer =
        document.querySelector('#real-container');
    inferenceContainer.innerHTML = '';
    inputNDArrayVisualizers = [];
    outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.
        // const ndarrayImageVisualizer =
        //     document.createElement('ndarray-image-visualizer');
        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement);
        ndarrayImageVisualizer.setShape(inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        inputNDArrayVisualizers.push(ndarrayImageVisualizer);
        // inferenceExampleElement.appendChild(ndarrayImageVisualizer);

        // Set up the output ndarray visualizer.
        const ndarrayLogitsVisualizer = new NDArrayLogitsVisualizer(inferenceExampleElement, 2);
        // const ndarrayLogitsVisualizer =
        //     document.createElement('ndarray-logits-visualizer');
        ndarrayLogitsVisualizer.initialize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
        // inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}

function buildFakeImageContainer() {
    const inferenceContainer =
        document.querySelector('#generated-container');
    inferenceContainer.innerHTML = '';
    fakeInputNDArrayVisualizers = [];
    fakeOutputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.
        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement)
        // document.createElement('ndarray-image-visualizer');
        ndarrayImageVisualizer.setShape(inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        fakeInputNDArrayVisualizers.push(ndarrayImageVisualizer);
        // inferenceExampleElement.appendChild(ndarrayImageVisualizer);

        // Set up the output ndarray visualizer.
        const ndarrayLogitsVisualizer = new NDArrayLogitsVisualizer(inferenceExampleElement, 2);
        // document.createElement('ndarray-logits-visualizer');
        ndarrayLogitsVisualizer.initialize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        fakeOutputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
        // inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}

function populateModelDropdown() {
    const _modelNames = ['Custom'];
    const _genModelNames = ['Custom'];

    const modelConfigs =
        xhrDatasetConfigs[selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
        if (modelConfigs.hasOwnProperty(modelName)) {
            if (modelName.endsWith('(disc)')) {
                _modelNames.push(modelName);
            } else {
                _genModelNames.push(modelName);
            }
        }
    }

    modelNames = _modelNames;
    genModelNames = _genModelNames;
    selectedModelName = modelNames[modelNames.length - 1];
    genSelectedModelName = genModelNames[genModelNames.length - 1];
    updateSelectedModel(selectedModelName, 'disc');
    updateSelectedModel(genSelectedModelName, 'gen');
}

function updateSelectedModel(modelName, which) {
    removeAllLayers(which);
    if (modelName === 'Custom') {
        // TODO(nsthorat): Remember the custom layers.
        return;
    }

    loadModelFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[modelName].path, which);
}

function loadModelFromPath(modelPath, which) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadModelFromJson(xhr.responseText, which);
    };
    xhr.onerror = (error) => {
        throw new Error(
            'Model could not be fetched from ' + modelPath + ': ' + error);
    };
    xhr.send();
}

function setupDatasetStats() {
    datasetStats = dataSet.getStats();
    statsExampleCount = datasetStats[IMAGE_DATA_INDEX].exampleCount;
    document.getElementById("statsExampleCount").innerHTML = `${statsExampleCount}`;
    statsInputRange = '[' + datasetStats[IMAGE_DATA_INDEX].inputMin +
        ', ' + datasetStats[IMAGE_DATA_INDEX].inputMax + ']';
    document.getElementById("statsInputRange").innerHTML = `${statsInputRange}`;
    statsInputShapeDisplay = getDisplayShape(
        datasetStats[IMAGE_DATA_INDEX].shape);
    document.getElementById("statsInputShapeDisplay").innerHTML = `${statsInputShapeDisplay}`;
    statsLabelShapeDisplay = getDisplayShape(
        datasetStats[LABEL_DATA_INDEX].shape);
    document.getElementById("statsLabelShapeDisplay").innerHTML = `${statsLabelShapeDisplay}`;
    showDatasetStats = true;
}

function applyNormalization(selectedNormalizationOption) {
    switch (selectedNormalizationOption) {
        case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                break;
            }
        case Normalization.NORMALIZATION_ZERO_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
                break;
            }
        case Normalization.NORMALIZATION_NONE:
            {
                dataSet.removeNormalization(IMAGE_DATA_INDEX);
                break;
            }
        default:
            {
                throw new Error('Normalization option must be 0, 1, or 2');
            }
    }
    setupDatasetStats();
}

// function recreateCharts() {
//     costChartData = [];
//     if (costChart != null) {
//         costChart.destroy();
//     }
//     costChart =
//         createChart('cost-chart', 'Discriminator Cost', costChartData, 0);

//     if (accuracyChart != null) {
//         accuracyChart.destroy();
//     }
//     accuracyChartData = [];
//     accuracyChart = createChart(
//         'accuracy-chart', 'Generator Cost', accuracyChartData, 0);

//     if (examplesPerSecChart != null) {
//         examplesPerSecChart.destroy();
//     }
//     examplesPerSecChartData = [];
//     examplesPerSecChart = createChart(
//         'examplespersec-chart', 'Examples/sec', examplesPerSecChartData,
//         0);
// }

// function createChart(
//     canvasId, label, data, min = null,
//     max = null) {
//     const context = (document.getElementById(canvasId)).getContext('2d');
//     return new Chart(context, {
//         type: 'line',
//         data: {
//             datasets: [{
//                 data,
//                 fill: false,
//                 label,
//                 pointRadius: 0,
//                 borderColor: 'rgba(75,192,192,1)',
//                 borderWidth: 1,
//                 lineTension: 0,
//                 pointHitRadius: 8
//             }]
//         },
//         options: {
//             animation: {
//                 duration: 0
//             },
//             responsive: false,
//             scales: {
//                 xAxes: [{
//                     type: 'linear',
//                     position: 'bottom'
//                 }],
//                 yAxes: [{
//                     ticks: {
//                         max,
//                         min,
//                     }
//                 }]
//             }
//         }
//     });
// }

function displayBatchesTrained(totalBatchesTrained) {
    examplesTrained = batchSize * totalBatchesTrained;
    document.getElementById("examplesTrained").innerHTML = `Examples trained: ${examplesTrained}`
}


var lossGraph = new cnnvis.Graph();
var lossWindow = new cnnutil.Window(100);

function displayCost(avgCost, which) {

    if (which === 'disc') {
        var cost = avgCost.get();
        var batchesTrained = graphRunner.getTotalBatchesTrained();

        lossWindow.add(cost);

        var xa = lossWindow.get_average();

        if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
            lossGraph.add(batchesTrained, xa);
            lossGraph.drawSelf(document.getElementById("lossgraph"));
        }
    } else {

        displayAccuracy(avgCost)
    }

}

// function displayCost(cost, which) {
//     if (which === 'disc') {
//         costChartData.push({
//             x: graphRunner.getTotalBatchesTrained(),
//             y: cost.get()
//         });
//         costChart.update();
//     } else {
//         accuracyChartData.push({
//             x: graphRunner.getTotalBatchesTrained(),
//             y: cost.get()
//         });
//         accuracyChart.update();
//     }
// }


var accuracyGraph = new cnnvis.Graph();
var accuracyWindow = new cnnutil.Window(100);

function displayAccuracy(accuracy) {

    var accuracy = accuracy.get() * 100;
    var batchesTrained = graphRunner.getTotalBatchesTrained();

    accuracyWindow.add(accuracy);

    var xa = accuracyWindow.get_average();

    if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
        accuracyGraph.add(batchesTrained, xa);
        accuracyGraph.drawSelf(document.getElementById("accuracygraph"));
    }
}

// function displayAccuracy(accuracy) {
//     accuracyChartData.push({
//         x: graphRunner.getTotalBatchesTrained(),
//         y: accuracy.get() * 100
//     });
//     accuracyChart.update();
// }

function displayInferenceExamplesPerSec(examplesPerSec) {
    inferencesPerSec =
        smoothExamplesPerSec(inferencesPerSec, examplesPerSec);
    inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));

    generationsPerSec = inferencesPerSec;
    generationDuration = inferenceDuration;

    document.getElementById("inferencesPerSec").innerHTML = `Inferences/sec: ${inferencesPerSec}`;
    document.getElementById("inferenceDuration").innerHTML = `inference Duration: ${inferenceDuration} ms`;

    document.getElementById("generationsPerSec").innerHTML = `Inferences/sec: ${generationsPerSec}`;
    document.getElementById("generationDuration").innerHTML = `inference Duration: ${generationDuration} ms`;
}


var examplesPerSecGraph = new cnnvis.Graph();
var examplesPerSecWindow = new cnnutil.Window(100);

function displayExamplesPerSec(_examplesPerSec) {


    var batchesTrained = graphRunner.getTotalBatchesTrained();

    examplesPerSecWindow.add(_examplesPerSec);

    var xa = examplesPerSecWindow.get_average();

    if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
        examplesPerSecGraph.add(batchesTrained, xa);
        examplesPerSecGraph.drawSelf(document.getElementById("examplespersecgraph"));
    }

    examplesPerSec =
        smoothExamplesPerSec(examplesPerSec, _examplesPerSec);

    document.getElementById("examplesPerSec").innerHTML = `Examples/sec: ${examplesPerSec}`;
}

// function displayExamplesPerSec(examplesPerSec) {
//     examplesPerSecChartData.push({
//         x: graphRunner.getTotalBatchesTrained(),
//         y: examplesPerSec
//     });
//     examplesPerSecChart.update();
//     examplesPerSec =
//         smoothExamplesPerSec(examplesPerSec, examplesPerSec);
// }

function smoothExamplesPerSec(
    lastExamplesPerSec, nextExamplesPerSec) {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
        .toPrecision(3));
}

function displayInferenceExamplesOutput(
    inputFeeds, inferenceOutputs) {

    let realImages = [];
    const realLabels = [];
    const realLogits = [];

    let fakeImages = [];
    const fakeLabels = [];
    const fakeLogits = [];

    for (let i = 0; i < inputFeeds.length; i++) {
        realImages.push(inputFeeds[i][0].data);
        realLabels.push(inputFeeds[i][2].data);
        realLogits.push(inferenceOutputs[2][i]);
        fakeImages.push((inferenceOutputs[0][i]));
        fakeLabels.push(inputFeeds[i][3].data);
        fakeLogits.push(inferenceOutputs[1][i]);
    }

    realImages =
        dataSet.unnormalizeExamples(realImages, IMAGE_DATA_INDEX);

    fakeImages =
        dataSet.unnormalizeExamples(fakeImages, IMAGE_DATA_INDEX);

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
        inputNDArrayVisualizers[i].saveImageDataFromNDArray(realImages[i]);
        fakeInputNDArrayVisualizers[i].saveImageDataFromNDArray(fakeImages[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
        const realSoftmaxLogits = math.softmax(realLogits[i]);
        const fakeSoftmaxLogits = math.softmax(fakeLogits[i]);

        outputNDArrayVisualizers[i].drawLogits(
            realSoftmaxLogits, realLabels[i]);
        fakeOutputNDArrayVisualizers[i].drawLogits(
            fakeSoftmaxLogits, fakeLabels[i]);
        inputNDArrayVisualizers[i].draw();
        fakeInputNDArrayVisualizers[i].draw();

        realSoftmaxLogits.dispose();
        fakeSoftmaxLogits.dispose();
    }
}

function addLayer(which) {

    const modelLayer = new ModelLayer(); //document.createElement('model-layer');

    if (which === 'gen') {

        const lastHiddenLayer = genHiddenLayers[genHiddenLayers.length - 1];
        const lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            randVectorShape;
        genHiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape, which);

        genLayersContainer.appendChild(modelLayer.paramContainer);
    } else {

        const lastHiddenLayer = discHiddenLayers[discHiddenLayers.length - 1];
        const lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            inputShape;
        discHiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape, which);

        layersContainer.appendChild(modelLayer.paramContainer);
    }



    return modelLayer;
}

function removeLayer(modelLayer, which) {
    if (which === 'gen') {
        genLayersContainer.removeChild(modelLayer.paramContainer);
        genHiddenLayers.splice(genHiddenLayers.indexOf(modelLayer), 1);
    } else {
        layersContainer.removeChild(modelLayer.paramContainer);
        discHiddenLayers.splice(discHiddenLayers.indexOf(modelLayer), 1);
    }
    layerParamChanged();
}

function removeAllLayers(which) {
    if (which === 'gen') {
        for (let i = 0; i < genHiddenLayers.length; i++) {
            genLayersContainer.removeChild(genHiddenLayers[i].paramContainer);
        }
        genHiddenLayers = [];
    } else {
        for (let i = 0; i < discHiddenLayers.length; i++) {
            layersContainer.removeChild(discHiddenLayers[i].paramContainer);
        }
        discHiddenLayers = [];
    }

    layerParamChanged();
}

function validateModel() {
    let valid = true;
    for (let i = 0; i < discHiddenLayers.length; ++i) {
        valid = valid && discHiddenLayers[i].isValid();
    }
    if (discHiddenLayers.length > 0) {
        const lastLayer = discHiddenLayers[discHiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(labelShape, lastLayer.getOutputShape());
    }
    valid = valid && (discHiddenLayers.length > 0);

    for (let i = 0; i < genHiddenLayers.length; ++i) {
        valid = valid && genHiddenLayers[i].isValid();
    }
    if (genHiddenLayers.length > 0) {
        const lastLayer = genHiddenLayers[genHiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(inputShape, lastLayer.getOutputShape());
    }
    valid = valid && (genHiddenLayers.length > 0);

    isValid = valid;
}

function layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = inputShape;
    for (let i = 0; i < discHiddenLayers.length; i++) {
        lastOutputShape = discHiddenLayers[i].setInputShape(lastOutputShape);
    }

    lastOutputShape = randVectorShape;
    for (let i = 0; i < genHiddenLayers.length; i++) {
        lastOutputShape = genHiddenLayers[i].setInputShape(lastOutputShape);
    }

    validateModel();

    if (isValid) {
        createModel();
    }
}

function downloadModel() {
    const modelJson = getModelAsJson();
    const blob = new Blob([modelJson], {
        type: 'text/json'
    });
    const textFile = window.URL.createObjectURL(blob);

    // Force a download.
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = textFile;
    // tslint:disable-next-line:no-any
    (a).download = selectedDatasetName + '_model';
    a.click();

    document.body.removeChild(a);
    window.URL.revokeObjectURL(textFile);
}

function uploadModel() {
    (document.querySelector('#model-file')).click();
}

function setupUploadModelButton() {
    // Show and setup the load view button.
    const fileInput = document.querySelector('#model-file');
    fileInput.addEventListener('change', event => {
        const file = fileInput.files[0];
        // Clear out the value of the file chooser. This ensures that if the user
        // selects the same file, we'll re-read it.
        fileInput.value = '';
        const fileReader = new FileReader();
        fileReader.onload = (evt) => {
            removeAllLayers('disc');
            const modelJson = fileReader.result;
            loadModelFromJson(modelJson, 'disc');
        };
        fileReader.readAsText(file);
    });
}

function getModelAsJson() {
    const layerBuilders = [];
    for (let i = 0; i < discHiddenLayers.length; i++) {
        layerBuilders.push(discHiddenLayers[i].layerBuilder);
    }
    return JSON.stringify(layerBuilders);
}

function loadModelFromJson(modelJson, which) {
    var lastOutputShape;
    var hiddenLayers;
    if (which === 'disc') {
        lastOutputShape = inputShape;
        hiddenLayers = discHiddenLayers;
    } else {
        lastOutputShape = randVectorShape;
        hiddenLayers = genHiddenLayers;
    }

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = addLayer(which);
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);
        insertLayerTableRow(modelLayer.paramContainer,
            modelLayer.selectedLayerName,
            modelLayer.inputShapeDisplay,
            modelLayer.outputShapeDisplay)
    }
    validateModel();
}

function uploadWeights() {
    (document.querySelector('#weights-file')).click();
}

function setupUploadWeightsButton(fileInput) {
    // Show and setup the load view button.
    // const fileInput = document.querySelector('#weights-file');
    fileInput.addEventListener('change', event => {
        const file = fileInput.files[0];
        // Clear out the value of the file chooser. This ensures that if the user
        // selects the same file, we'll re-read it.
        fileInput.value = '';
        const fileReader = new FileReader();
        fileReader.onload = (evt) => {
            const weightsJson = fileReader.result;
            loadedWeights = JSON.parse(weightsJson);
            createModel();
        };
        fileReader.readAsText(file);
    });
}

// method 2
function getWeightsFromGraph(graph, name = 'generator') {

    let weights = {};

    genNodes = graph.getNodes().filter((x) =>
        x.name.startsWith(name));

    for (let j = 0; j < genNodes.length; j++) {

        genNodes[j].data.dataSync();

        values = genNodes[j].data;

        console.log('genNodes', j, genNodes[j].name, values)

        weights[genNodes[j].name] = values;

    }

    return weights // only those layerweights with non-null data, w,b unpacked [w,b,w,b,...]
}

function downloadWeightsAsJson(weights) {

    var json_text = JSON.stringify(weights);

    const blob = new Blob([json_text], {
        type: 'text/json'
    });
    const textFile = window.URL.createObjectURL(blob);

    // Force a download.
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = textFile;
    // tslint:disable-next-line:no-any
    (a).download = './generatorWeights.json';
    a.click();

    document.body.removeChild(a);
    window.URL.revokeObjectURL(textFile);
}


function run() {

    discLearningRate = 0.01;
    genLearningRate = 0.01;
    discMomentum = 0.1;
    genMomentum = 0.1;
    discNeedMomentum = false;
    genNeedMomentum = false;
    discGamma = 0.1;
    genGamma = 0.1;
    discBeta1 = 0.9;
    discBeta2 = 0.999;
    genBeta1 = 0.9;
    genBeta2 = 0.999;
    discNeedGamma = false;
    genNeedGamma = false;
    discNeedBeta = false;
    genNeedBeta = true;
    batchSize = 15;

    updateNetParamDisplay();

    var normalizationDropdown = document.getElementById("normalization-dropdown");
    normalizationDropdown.options[selectedNormalizationOption].selected = 'selected';

    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName, graphRunner);


    // Default optimizer is momentum
    discSelectedOptimizerName = "sgd";
    genSelectedOptimizerName = "adam";

    var discOptimizerDropdown = document.getElementById("disc-optimizer-dropdown");
    var ind = indexOfDropdownOptions(discOptimizerDropdown.options, discSelectedOptimizerName)
    discOptimizerDropdown.options[ind].selected = 'selected';

    var genOptimizerDropdown = document.getElementById("gen-optimizer-dropdown");
    var ind = indexOfDropdownOptions(genOptimizerDropdown.options, genSelectedOptimizerName)
    genOptimizerDropdown.options[ind].selected = 'selected';

    const eventObserver = {
        batchesTrainedCallback: (batchesTrained) =>
            displayBatchesTrained(batchesTrained),
        discCostCallback: (cost) => displayCost(cost, 'disc'),
        genCostCallback: (cost) => displayCost(cost, 'gen'),
        metricCallback: (metric) => displayAccuracy(metric),
        inferenceExamplesCallback:
            (inputFeeds, inferenceOutputs) =>
            displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
        //        console.log(inputFeeds, inferenceOutputs),
        inferenceExamplesPerSecCallback: (examplesPerSec) =>
            displayInferenceExamplesPerSec(examplesPerSec),
        trainExamplesPerSecCallback: (examplesPerSec) =>
            displayExamplesPerSec(examplesPerSec),
        totalTimeCallback: (totalTimeSec) => {
            totalTimeSec = totalTimeSec.toFixed(1);
            document.getElementById("totalTimeSec").innerHTML = `Total time: ${totalTimeSec} sec.`;
        },
    };
    graphRunner = new MyGraphRunner(math, session, eventObserver);


    applicationState = ApplicationState.IDLE;
    loadedWeights = null;
    modelInitialized = false;
    showTrainStats = false;
    showDatasetStats = false;

    discHiddenLayers = [];
    genHiddenLayers = [];
    examplesPerSec = 0;
    inferencesPerSec = 0;
    generationsPerSec = 0;
    randVectorShape = [100];

    // Set up datasets.
    populateDatasets();
    // createModel();



    document.querySelector('#dataset-dropdown')
        .addEventListener(
            'change', (event) => {

                if (graphRunner != null) {
                    graphRunner = null;
                }
                graphRunner = new MyGraphRunner(math, session, eventObserver);

                // Update the dataset.
                const datasetName = event.target.value;
                updateSelectedDataset(datasetName);

                // TODO(nsthorat): Remember the last model used for each dataset.
                removeAllLayers('gen');
                removeAllLayers('disc');
            });


    document.querySelector('#model-dropdown').addEventListener(
        'change', (event) => {
            // Update the model.

            const modelName = event.target.value;
            updateSelectedModel(modelName, 'disc');
            console.log('dis model =', modelName)
        });


    document.querySelector('#gen-model-dropdown').addEventListener(
        'change', (event) => {
            // Update the model.

            const modelName = event.target.value;
            updateSelectedModel(modelName, 'gen');
            console.log('gen model =', modelName)
        });

    {
        const normalizationDropdown =
            document.querySelector('#normalization-dropdown');
        normalizationDropdown.addEventListener('change', (event) => {
            const selectedNormalizationOption = Number(event.target.value);

            console.log('normalization =', event.target.options[selectedNormalizationOption].innerHTML);
            applyNormalization(selectedNormalizationOption);
            setupDatasetStats();
        });
    }


    document.querySelector('#disc-optimizer-dropdown').addEventListener('change', (event) => {
        // Activate, deactivate hyper parameter inputs.
        refreshHyperParamRequirements(event.target.value, 'disc');
        discSelectedOptimizerName = event.target.value;
        console.log('disc optimizer =', event.target.value)
    });



    document.querySelector('#gen-optimizer-dropdown').addEventListener('change', (event) => {
        // Activate, deactivate hyper parameter inputs.
        refreshHyperParamRequirements(event.target.value, 'gen');
        genSelectedOptimizerName = event.target.value;
        console.log('gen optimizer =', event.target.value)
    });





    // const addButton = document.querySelector('#add-layer');
    // addButton.addEventListener('click', () => addLayer('disc'));

    // const genAddButton = document.querySelector('#gen-add-layer');
    // genAddButton.addEventListener('click', () => addLayer('gen'));

    /*
        const downloadModelButton = document.querySelector('#download-model');
        downloadModelButton.addEventListener('click', () => downloadModel());
        const uploadModelButton = document.querySelector('#upload-model');
        uploadModelButton.addEventListener('click', () => uploadModel());
        setupUploadModelButton();
    */

    const downloadWeightsButton = document.querySelector('#btnweights');
    downloadWeightsButton.addEventListener('click', () => {
        let weights = getWeightsFromGraph(graph)
        // let weights = getWeightsFromHiddenLayers(genHiddenLayers);
        downloadWeightsAsJson(weights);
    });
    const fileInput = document.querySelector('#weights-file');
    setupUploadWeightsButton(fileInput);

    document.querySelector('#environment-dropdown').addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName, graphRunner)
    });

}


function updateSelectedEnvironment(selectedEnvName, _graphRunner = null) {

    math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
    console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU')
    if (_graphRunner != null) {
        _graphRunner.setMath(math);
    }

}


var updateNetParamDisplay = function () {
    document.getElementById('disc-learning-rate-input').value = discLearningRate;
    document.getElementById('disc-momentum').value = discMomentum;
    document.getElementById('gen-learning-rate-input').value = genLearningRate;
    document.getElementById('gen-momentum').value = genMomentum;

    document.getElementById('batch_size_input').value = batchSize;
    // document.getElementById('decay_input').value = trainer.l2_decay;
}


// user settings
var changeNetParam = function () {

    discLearningRate = parseFloat(document.getElementById("disc-learning-rate-input").value);
    if (graphRunner.discOptimizer != null && discLearningRate !== graphRunner.discOptimizer.learningRate) {
        graphRunner.discOptimizer.learningRate = discLearningRate;

        console.log('disc learning rate changed to' + discLearningRate);
    }

    discMomentum = parseFloat(document.getElementById("disc-momentum").value);
    if (graphRunner.discOptimizer != null && discMomentum !== graphRunner.discOptimizer.momentum) {
        graphRunner.discOptimizer.momentum = discMomentum;

        console.log('disc momentum changed to' + discMomentum);
    }

    genLearningRate = parseFloat(document.getElementById("gen-learning-rate-input").value);
    if (graphRunner.genOptimizer != null && genLearningRate !== graphRunner.genOptimizer.learningRate) {
        graphRunner.genOptimizer.learningRate = genLearningRate;

        console.log('gen learning rate changed to' + genLearningRate);
    }

    genMomentum = parseFloat(document.getElementById("gen-momentum").value);
    if (graphRunner.genOptimizer != null && genMomentum !== graphRunner.genOptimizer.momentum) {
        graphRunner.genOptimizer.momentum = genMomentum;

        console.log('gen momentum changed to' + genMomentum);
    }

    batchSize = parseFloat(document.getElementById("batch_size_input").value);
    if (graphRunner.batchSize != null && batchSize != graphRunner.batchSize) {
        graphRunner.batchSize = batchSize;
        console.log('batch size changed to' + batchSize);
    }

    updateNetParamDisplay();
}

var infer_request = null;
var btn_infer = document.getElementById('buttoninfer');
var infer_paused = true;
btn_infer.addEventListener('click', () => {

    infer_paused = !infer_paused;
    if (infer_paused) {
        btn_infer.value = 'Start Inferring';
        if (graphRunner != null) {
            graphRunner.stopInferring();
        }


    } else {

        infer_request = true;
        btn_infer.value = 'Pause Inferring';


    }
});

var train_request = null;
var btn_train = document.getElementById('buttontrain');
var train_paused = true;
btn_train.addEventListener('click', () => {
    train_paused = !train_paused;

    if (train_paused) {
        if (graphRunner != null) {
            graphRunner.stopTraining();
        }
        // btn_train.value = 'Start Training';

    } else {

        train_request = true;

        // btn_train.value = 'Pause Training';

    }
});


function monitor() {

    if (modelInitialized == false) {

        btn_infer.disabled = true;
        btn_infer.value = 'Initializing Model ...'
        // btn_train.disabled = true;
        btn_train.style.visibility = 'hidden';

    } else {
        if (isValid) {

            btn_infer.className = 'btn btn-primary btn-md';
            btn_infer.disabled = false;
            btn_train.style.visibility = 'visible';

            if (infer_paused) {
                btn_infer.value = 'Start Infering'
            } else {
                btn_infer.value = 'Stop Infering'
            }

            if (train_paused) {
                if (graphRunner.getTotalBatchesTrained() > 0) {
                    btn_train.value = 'Resume Training'
                } else {
                    btn_train.value = 'Start Training'
                }
            } else {
                btn_train.value = 'Stop Training'
            }


            if (train_request) {
                train_request = false;
                // createModel();
                if (graphRunner.getTotalBatchesTrained() > 0) {
                    graphRunner.resumeTraining()
                } else {
                    examplesPerSecGraph.pts = [];
                    accuracyGraph.pts = [];
                    lossGraph.pts = [];

                    startTraining();
                }

            }

            if (infer_request) {
                infer_request = false;
                // createModel();
                startInference();
            }

        } else {
            btn_infer.className = 'btn btn-danger btn-md';
            btn_infer.disabled = true;
            btn_infer.value = 'Model not valid or being reinitialized'
            // btn_train.disabled = true;
            btn_train.style.visibility = 'hidden';
        }
    }

    setTimeout(function () {
        monitor();
    }, 100);
}

function start() {

    supported = detect_support();

    if (supported) {
        console.log('device & webgl supported');
        btn_infer.disabled = false;
        btn_train.disabled = false;

        setTimeout(function () {

            run();

            monitor();

        }, 0);

    } else {
        console.log('device/webgl not supported')
        btn_infer.disabled = true;
        btn_train.disabled = true;
    }



}