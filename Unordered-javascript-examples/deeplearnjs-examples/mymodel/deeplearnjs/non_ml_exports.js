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


const graph = new Graph();
// Make a new input in the graph, called 'x', with shape [] (a Scalar).
const x = graph.placeholder('x', []);
// Make new variables in the graph, 'a', 'b', 'c' with shape [] and random
// initial values.
const a = graph.variable('a', Scalar.new(Math.random()));
const b = graph.variable('b', Scalar.new(Math.random()));
const c = graph.variable('c', Scalar.new(Math.random()));
// Make new tensors representing the output of the operations of the quadratic.
const order2 = graph.multiply(a, graph.square(x));
const order1 = graph.multiply(b, x);
const y = graph.add(graph.add(order2, order1), c);

// When training, we need to provide a label and a cost function.
const yLabel = graph.placeholder('y label', []);
// Provide a mean squared cost function for training. cost = (y - yLabel)^2
const cost = graph.meanSquaredCost(y, yLabel);

// At this point the graph is set up, but has not yet been evaluated.
// **deeplearn.js** needs a Session object to evaluate a graph.
const math = new NDArrayMathGPU();
const session = new Session(graph, math);

// For more information on scope / track, check out the [tutorial on performance](/docs/tutorials/performance.html).
math.scope((keep, track) => {
    /**
     * Inference
     */
    // Now we ask the graph to evaluate (infer) and give us the result when
    // providing a value 4 for "x".
    // NOTE: "a", "b", and "c" are randomly initialized, so this will give us
    // something random.
    let result = session.eval(y, [{
        tensor: x,
        data: track(Scalar.new(4))
    }]);
    console.log(result.shape);
    //   console.log('result', await result.data());
    //   result.data().then(data => console.log(data));
    console.log(result.getValues());

    /**
     * Training
     */
    // Now let's learn the coefficients of this quadratic given some data.
    // To do this, we need to provide examples of x and y.
    // The values given here are for values a = 3, b = 2, c = 1, with random
    // noise added to the output so it's not a perfect fit.
    const xs = [
        track(Scalar.new(0)),
        track(Scalar.new(1)),
        track(Scalar.new(2)),
        track(Scalar.new(3))
    ];
    const ys = [
        track(Scalar.new(1.1)),
        track(Scalar.new(5.9)),
        track(Scalar.new(16.8)),
        track(Scalar.new(33.9))
    ];
    // When training, it's important to shuffle your data!
    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
    const [xProvider, yProvider] =
    shuffledInputProviderBuilder.getInputProviders();

    // Training is broken up into batches.
    const NUM_BATCHES = 20;
    const BATCH_SIZE = xs.length;
    // Before we start training, we need to provide an optimizer. This is the
    // object that is responsible for updating weights. The learning rate param
    // is a value that represents how large of a step to make when updating
    // weights. If this is too big, you may overstep and oscillate. If it is too
    // small, the model may take a long time to train.
    const LEARNING_RATE = .01;
    const optimizer = new SGDOptimizer(LEARNING_RATE);
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize; this call trains one batch and
        // returns the average cost of the batch as a Scalar.
        const costValue = session.train(
            cost,
            // Map input providers to Tensors on the graph.
            [{
                tensor: x,
                data: xProvider
            }, {
                tensor: yLabel,
                data: yProvider
            }],
            BATCH_SIZE, optimizer, CostReduction.MEAN);

        console.log('average cost: ' + costValue.getValues());

        // console.log(result.getValues());
    }

    // Now print the value from the trained model for x = 4, should be ~57.0.
    result = session.eval(y, [{
        tensor: x,
        data: track(Scalar.new(4))
    }]);
    console.log('result should be ~57.0:');
    console.log(result.shape);
    //   console.log(await result.data());
    // result.data().then(data => console.log(data));
    console.log(result.getValues());
});