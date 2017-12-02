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

var dl = deeplearn;
var Graph = dl.Graph;
var Tensor = dl.Tensor;
var Scalar = dl.Scalar;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Session = dl.Session;
var track = dl.track;
var keep = dl.keep;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var SGDOptimizer = dl.SGDOptimizer;
var CostReduction = dl.CostReduction;
var Array1D = dl.Array1D;



/**
 * This implementation of computing the complementary color came from an
 * answer by Edd https://stackoverflow.com/a/37657940
 */
function computeComplementaryColor(rgbColor) {
    let r = rgbColor[0];
    let g = rgbColor[1];
    let b = rgbColor[2];

    // Convert RGB to HSL
    // Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
    r /= 255.0;
    g /= 255.0;
    b /= 255.0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = (max + min) / 2.0;
    let s = h;
    const l = h;

    if (max === min) {
        h = s = 0; // achromatic
    } else {
        const d = max - min;
        s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));

        if (max === r && g >= b) {
            h = 1.0472 * (g - b) / d;
        } else if (max === r && g < b) {
            h = 1.0472 * (g - b) / d + 6.2832;
        } else if (max === g) {
            h = 1.0472 * (b - r) / d + 2.0944;
        } else if (max === b) {
            h = 1.0472 * (r - g) / d + 4.1888;
        }
    }

    h = h / 6.2832 * 360.0 + 0;

    // Shift hue to opposite side of wheel and convert to [0-1] value
    h += 180;
    if (h > 360) {
        h -= 360;
    }
    h /= 360;

    // Convert h s and l values into r g and b values
    // Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
    if (s === 0) {
        r = g = b = l; // achromatic
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;

        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }

    return [r, g, b].map(v => Math.round(v * 255));
}


function generateRandomChannelValue() {
    return Math.floor(Math.random() * 256);
}

function normalizeColor(rgbColor) {
    return rgbColor.map(v => v / 255);
}

function denormalizeColor(normalizedRgbColor) {
    return normalizedRgbColor.map(v => v * 255);
}




function createFullyConnectedLayer(
    graph, inputLayer, layerIndex,
    sizeOfThisLayer, includeRelu = true, includeBias = true) {
    return graph.layers.dense(
        'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
        includeRelu ? (x) => graph.relu(x) : undefined, includeBias);
}


//////////////////////
///Train and Predict//
//////////////////////


function injectNoise(feedEntries, costTensor) {

    class FeedDictionary {
        /**
         * Optionally construct a FeedDictionary from an array of entries.
         * @param feedEntries Optional array of FeedEntry objects.
         */
        constructor(feedEntries = null) {
            this.dict = {};
            if (feedEntries) {
                feedEntries.forEach(entry => this.dict[entry.tensor.id] = entry);
            }
        }
    }

    class VariableNode extends Node {
        constructor(graph, name, data) {
            super(graph, name, {}, new Tensor(data.shape));
        }
        validate() {
            util.assert(
                this.data != null,
                'Error adding variable op: Data for variable \'' + this.name +
                '\' is null or undefined');
        }
    }

    function getVariableNodesFromEvaluationSet(evaluationSet) {
        const nodes = [];
        evaluationSet.forEach(node => {
            if (node instanceof VariableNode) {
                nodes.push(node);
            }
        });
        return nodes;
    }

    var feed = new FeedDictionary(feedEntries);
    var runtime = session.getOrCreateRuntime([costTensor], feed);
    var variableNodes = optimizer.specifiedVariableNodes == null ?
        getVariableNodesFromEvaluationSet(runtime.nodes) :
        this.specifiedVariableNodes;

    variableNodes.forEach(node => {
        const oldVariable = session.activationArrayMap.get(node.output);
        // const gradient = this.variableGradients.get(node.output);
        const noise = NDArray.randUniform(node.output.shape, -1 * noiseScale, 1 * noiseScale); // min max

        const variable = math.scaledArrayAdd(this.one, noise, this.one, oldVariable);
        session.activationArrayMap.set(node.output, keep(variable));
        node.data = variable;

        oldVariable.dispose();
    });
}

function train1Batch(shouldFetchCost, shouldInjectNoise) {
    // Every 42 steps, lower the learning rate by 15%.
    const learningRate =
        initialLearningRate * Math.pow(0.95, Math.floor(step / 82));
    optimizer.setLearningRate(learningRate);

    // Train 1 batch.
    let costValue = -1;
    math.scope(() => {
        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer,
            shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);

        if (shouldInjectNoise) {
            injectNoise(feedEntries, costTensor);
            //inject only once
            shouldInjectNoise = false;
        }

        if (!shouldFetchCost) {
            // We only train. We do not compute the cost.
            return;
        }

        // Compute the cost (by calling get), which requires transferring data
        // from the GPU.
        costValue = cost.get();
    });

    return costValue;
}

function predict(rgbColor) {
    let complementColor = [];
    math.scope((keep, track) => {
        const mapping = [{
            tensor: inputTensor,
            data: Array1D.new(normalizeColor(rgbColor)),
        }];
        const evalOutput = session.eval(predictionTensor, mapping);
        const values = evalOutput.getValues();
        const colors = denormalizeColor(Array.prototype.slice.call(values));

        // Make sure the values are within range.
        complementColor = colors.map(
            v => Math.round(Math.max(Math.min(v, 255), 0)));
    });
    return complementColor;
}


function populateContainerWithColor(
    container, r, g, b) {
    const originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
    container.textContent = originalColorString;

    const colorBox = document.createElement('div');
    colorBox.classList.add('color-box');
    colorBox.style.background = originalColorString;
    container.appendChild(colorBox);
}

var updateNetParamDisplay = function () {
    document.getElementById('learning-rate-input').value = initialLearningRate;
    document.getElementById('noise-scale-input').value = noiseScale;
    document.getElementById('egdiv').innerHTML = 'step = ' + step;
    // document.getElementById('batch_size_input').value = batchSize;
    // // document.getElementById('decay_input').value = trainer.l2_decay;
}

// user settings
var changeNetParam = function () {
    if (noiseScale !== parseFloat(document.getElementById("noise-scale-input").value)) {
        noiseScale = parseFloat(document.getElementById("noise-scale-input").value);

        console.log('noise scale changed to' + noiseScale);
    }
    initialLearningRate = parseFloat(document.getElementById("learning-rate-input").value);
    if (optimizer != null && initialLearningRate !== optimizer.learningRate) {
        optimizer.learningRate = initialLearningRate;

        console.log('learning rate changed to' + initialLearningRate);
    }
    updateNetParamDisplay();
}




function initializeUi() {
    const colorRows = document.querySelectorAll('tr[data-original-color]');
    for (let i = 0; i < colorRows.length; i++) {
        const rowElement = colorRows[i];
        const tds = rowElement.querySelectorAll('td');
        const originalColor =
            (rowElement.getAttribute('data-original-color'))
            .split(',')
            .map(v => parseInt(v, 10));

        // Visualize the original color.
        populateContainerWithColor(
            tds[0], originalColor[0], originalColor[1], originalColor[2]);

        // Visualize the complementary color.
        const complement =
            computeComplementaryColor(originalColor);
        populateContainerWithColor(
            tds[1], complement[0], complement[1], complement[2]);
    }

    UI_initialized = true
}


function make_plot_responsive() {

    (function () {
        var d3 = Plotly.d3;
        var WIDTH_IN_PERCENT_OF_PARENT = 100,
            HEIGHT_IN_PERCENT_OF_PARENT = 90;
        var gd3 = d3.selectAll(".responsive-plot")
            .style({
                width: WIDTH_IN_PERCENT_OF_PARENT + '%',
                'margin-left': (100 - WIDTH_IN_PERCENT_OF_PARENT) / 2 + '%',
                height: HEIGHT_IN_PERCENT_OF_PARENT + 'vh',
                'margin-top': (100 - HEIGHT_IN_PERCENT_OF_PARENT) / 2 + 'vh'
            });
        var nodes_to_resize = gd3[0]; //not sure why but the goods are within a nested array

        function resize_plot() {
            for (var i = 0; i < nodes_to_resize.length; i++) {
                Plotly.Plots.resize(nodes_to_resize[i]);
            }
        }
        resize_plot();
        window.onresize = resize_plot;
    })();

}

function create_plot3d(init_x, init_y, init_z) {

    Plotly.d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/3d-line1.csv', function (err, rows) {

        var data = [{
            type: 'scatter3d',
            mode: 'lines',
            x: init_x,
            y: init_y,
            z: init_z,
            opacity: 1,
            line: {
                width: 5,
                // color: c,
                reversescale: false
            },
            displayModeBar: false
        }];
        var layout = {


            scene: {
                xaxis: {
                    nticks: 5,
                    title: 'R',
                    backgroundcolor: "rgb(200, 200, 230)",
                    gridcolor: "rgb(255, 255, 255)",
                    showbackground: true,
                    zerolinecolor: "rgb(255, 255, 255)",
                },
                yaxis: {
                    nticks: 5,
                    title: 'G',
                    backgroundcolor: "rgb(230, 200,230)",
                    gridcolor: "rgb(255, 255, 255)",
                    showbackground: true,
                    zerolinecolor: "rgb(255, 255, 255)"
                },
                zaxis: {
                    nticks: 5,
                    title: 'B',
                    backgroundcolor: "rgb(230, 230,200)",
                    gridcolor: "rgb(255, 255, 255)",
                    showbackground: true,
                    zerolinecolor: "rgb(255, 255, 255)"
                }
            },
            autosize: true,
            // height: 300,
            margin: {
                l: 2,
                r: 2,
                b: 2,
                t: 2,
                pad: 0
            },
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#c7c7c7'
        };

        Plotly.newPlot('graph-color', data, layout);

        make_plot_responsive()

    });

}

function update_plot3d(new_x, new_y, new_z) {

    Plotly.d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/3d-line1.csv', function (err, rows) {
        Plotly.animate('graph-color', {
            data: [{
                x: new_x,
                y: new_y,
                z: new_z
            }],
            traces: [0],
            layout: {}
        }, {
            transition: {
                duration: 300,
                easing: 'cubic-in-out'
            }
        })

    });

}


function createChart(canvasId, label, data, min = 0, max = null) {

    const canvas = document.getElementById(canvasId);

    const context = canvas.getContext('2d');

    config.data.datasets[0].data = data;
    config.data.datasets[0].label = label;

    return new Chart(context, config);


}



function train_per() {
    if (step > 4242) {
        // Stop training.
        return;
    }

    if (paused) return;

    // We only fetch the cost every 10 steps because doing so requires a transfer
    // of data from the GPU.

    cost = train1Batch(step % 10 === 0, shouldInjectNoise);

    var d = document.getElementById('egdiv');
    d.innerHTML = 'step = ' + step;

    if (step % 10 === 0) {

        chartData.push({
            x: step,
            y: cost
        });

        config.data.datasets[0].data = chartData;
        chart.update();

        // Print data to console so the user can inspect.
        console.log('step', step - 1, 'cost', cost);

        // Visualize the predicted complement.
        const colorRows = document.querySelectorAll('tr[data-original-color]');
        for (let i = 0; i < colorRows.length; i++) {
            const rowElement = colorRows[i];
            const tds = rowElement.querySelectorAll('td');
            const originalColor =
                (rowElement.getAttribute('data-original-color'))
                .split(',')
                .map(v => parseInt(v, 10));

            // Visualize the predicted color.
            const predictedColor = predict(originalColor);
            populateContainerWithColor(
                tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);

            if (i === 0) {
                data_x.push(predictedColor[0])
                data_y.push(predictedColor[1])
                data_z.push(predictedColor[2])
                if (plot_exist === false) {
                    create_plot3d(data_x, data_y, data_z);
                    plot_exist = true
                } else {
                    update_plot3d(data_x, data_y, data_z);
                }
            }


        }
    }

    step++;
}



function toggle_pause() {
    paused = !paused;
    var btn = document.getElementById('buttontp');
    if (paused) {
        btn.value = 'Resume'
    } else {
        btn.value = 'Pause';
    }
}


function run() {

    updateNetParamDisplay();

    if (UI_initialized) {
        console.log('starting!');
        setInterval(train_per, 5); // lets go!
    } else {
        console.log('waiting!');
        setTimeout(run, 1000); // run again after 1second
    } // keep checking
}



var chartData;
var chart;
var chart_exist;

var paused;

// On every frame, we train and then maybe update the UI.
var step;

var plot_exist;

var data_x;
var data_y;
var data_z;

var UI_initialized;


var rawInputs;


var math;

var graph;

// This tensor contains the input. In this case, it is a scalar.
var inputTensor;

// This tensor contains the target.
var targetTensor;

var inputArray;
var targetArray;

var shuffledInputProviderBuilder;
var inputProvider;
var targetProvider;

var feedEntries;

// Create 3 fully connected layers, each with half the number of nodes of
// the previous layer. The first one has 16 nodes.
var fullyConnectedLayer;

var predictionTensor;

var costTensor;

var session;

var batchSize;
var initialLearningRate;
var optimizer;

var learningRateBtn;


// helper functions and variables to inject noise
var noiseScale;
var shouldInjectNoise;
var injectNoiseBtn;


var config = {
    type: 'line',
    data: {
        datasets: [{
            data: [],
            fill: false,
            label: ' ',
            // pointRadius: 0,
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
        responsive: false,
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

function start() {

    supported = detect_support();

    if (supported) {
        console.log('device & webgl supported')
        document.getElementById("buttontp").disabled = false;
    } else {
        console.log('device/webgl not supported')
        document.getElementById("buttontp").disabled = true;
    }

    UI_initialized = false
    initializeUi();

    chartData = []

    chart = createChart('plot', 'cost', chartData, 0, chartData.y);
    chart.update();
    chart_exist = true;

    paused = true;

    // On every frame, we train and then maybe update the UI.
    step = 0;

    plot_exist = false

    data_x = [];
    data_y = [];
    data_z = [];

    rawInputs = new Array(1e5);
    for (let i = 0; i < 1e5; i++) {
        rawInputs[i] = [
            generateRandomChannelValue(), generateRandomChannelValue(),
            generateRandomChannelValue()
        ];
    }


    math = new NDArrayMathGPU();

    graph = new Graph();

    // This tensor contains the input. In this case, it is a scalar.
    inputTensor = graph.placeholder('input RGB value', [3]);

    // This tensor contains the target.
    targetTensor = graph.placeholder('output RGB value', [3]);

    inputArray =
        rawInputs.map(c => Array1D.new(normalizeColor(c)));
    targetArray = rawInputs.map(
        c => Array1D.new(
            normalizeColor(computeComplementaryColor(c))));


    shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
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

    // Create 3 fully connected layers, each with half the number of nodes of
    // the previous layer. The first one has 16 nodes.
    fullyConnectedLayer =
        createFullyConnectedLayer(graph, inputTensor, 0, 64);

    // Create fully connected layer 1, which has 8 nodes.
    fullyConnectedLayer =
        createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);

    // Create fully connected layer 2, which has 4 nodes.
    fullyConnectedLayer =
        createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);


    predictionTensor =
        createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);

    costTensor =
        graph.meanSquaredCost(targetTensor, predictionTensor);

    session = new Session(graph, math);

    batchSize = 100;
    initialLearningRate = 0.5;
    optimizer = new SGDOptimizer(initialLearningRate);

    learningRateBtn = document.getElementById("learningRateBtn");
    learningRateBtn.addEventListener('click', () => {
        // Activate, deactivate hyper parameter inputs.
        changeNetParam();
    });


    // helper functions and variables to inject noise
    noiseScale = 0.3;
    shouldInjectNoise = false;
    injectNoiseBtn = document.getElementById("injectNoiseBtn");
    injectNoiseBtn.addEventListener('click', () => {
        // Activate, deactivate hyper parameter inputs.
        shouldInjectNoise = true;
        changeNetParam();
        console.log(`injected noise once with scale ${noiseScale}`);
    });


    run();

}