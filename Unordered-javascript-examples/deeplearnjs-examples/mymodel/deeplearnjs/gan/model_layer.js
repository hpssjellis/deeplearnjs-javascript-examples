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



// import {PolymerElement, PolymerHTMLElement} from './polymer-spec';

// import * as layer_builder from './layer_builder';
// import {LayerBuilder, LayerName, LayerWeightsDict} from './layer_builder';
// import {GANPlayground} from './gan-playground';
// import * as model_builder_util from './model_builder_util';

// tslint:disable-next-line:variable-name
// export let ModelLayerPolymer: new() => PolymerHTMLElement = PolymerElement({
//     is: 'model-layer',
//     properties: {
//         layerName: String,
//         inputShapeDisplay: String,
//         outputShapeDisplay: String,
//         isStatic: {
//             type: Boolean,
//             value: false
//         },
//         layerNames: Array,
//         selectedLayerName: String,
//         hasError: {
//             type: Boolean,
//             value: false
//         },
//         errorMessages: Array,
//     }
// });

class ModelLayer {

    constructor() {

        this.inputShapeDisplay = null;
        this.outputShapeDisplay = null;
        this.layerNames = null;
        this.selectedLayerName = null;
        this.hasError = null;
        this.errorMessages = null;
        this.ganPlayground = null;
        this.layerBuilder = null;
        this.inputShape = null;
        this.outputShape = null;
        this.paramContainer = null;
    }

    initialize(ganPlayground, inputShape,
        which) {
        this.ganPlayground = ganPlayground;
        this.paramContainer = document.createElement('div')
        // this.querySelector('.param-container');
        this.layerNames = [
            'Fully connected', 'ReLU', 'Convolution', 'Max pool', 'Reshape', 'Flatten'
        ];
        this.inputShape = inputShape;
        this.buildParamsUI('Fully connected', this.inputShape);

        // this.querySelector('.dropdown-content')
        //     .addEventListener(
        //         // tslint:disable-next-line:no-any
        //         'iron-activate', (event) => {
        //             this.buildParamsUI(
        //                 event.detail.selected, this.inputShape);
        //         });

        // this.querySelector('#remove-layer').addEventListener('click', (event) => {
        //     ganPlayground.removeLayer(this, which);
        // });
    }

    setInputShape(shape) {
        this.inputShape = shape;
        this.inputShapeDisplay =
            getDisplayShape(this.inputShape);

        const errors = [];
        const validationErrors = this.layerBuilder.validate(this.inputShape);
        if (validationErrors != null) {
            for (let i = 0; i < validationErrors.length; i++) {
                errors.push('Error: ' + validationErrors[i]);
            }
        }

        try {
            this.outputShape = this.layerBuilder.getOutputShape(this.inputShape);
        } catch (e) {
            errors.push(e);
        }
        this.outputShapeDisplay =
            getDisplayShape(this.outputShape);

        if (errors.length > 0) {
            this.hasError = true;
            this.errorMessages = errors;
        } else {
            this.hasError = false;
            this.errorMessages = [];
        }

        return this.outputShape;
    }

    isValid() {
        return !this.hasError;
    }

    getOutputShape() {
        return this.outputShape;
    }

    // addLayer(
    //     g, network, index,
    //     weights) {
    //     return this.layerBuilder.addLayer(
    //         g, network, this.inputShape, index, weights);
    // }

    addLayerMultiple(
        g, networks, name,
        weights) {
        return this.layerBuilder.addLayerMultiple(
            g, networks, this.inputShape, name, weights);
    }

    /**
     * Build parameters for the UI for a given op type. This is called when the
     * op is added, and when the op type changes.
     */
    buildParamsUI(
        layerName, inputShape,
        layerBuilderJson = null) {
        this.selectedLayerName = layerName;

        this.layerBuilder =
            getLayerBuilder(layerName, layerBuilderJson);

        // Clear any existing parameters.
        this.paramContainer.innerHTML = '';

        // Add all the parameters to the UI.
        const layerParams = this.layerBuilder.getLayerParams();
        for (let i = 0; i < layerParams.length; i++) {
            const initialValue = layerBuilderJson != null ?
                layerParams[i].getValue() :
                layerParams[i].initialValue(inputShape);
            this.addParamField(
                layerParams[i].label, initialValue, layerParams[i].setValue,
                layerParams[i].type, layerParams[i].min, layerParams[i].max);
        }
        this.ganPlayground.layerParamChanged();
    }

    loadParamsFromLayerBuilder(
        inputShape, layerBuilderJson) {
        this.buildParamsUI(
            layerBuilderJson.layerName, inputShape, layerBuilderJson);
    }

    addParamField(
        label, initialValue,
        setValue, type,
        min = null, max = null) {
        const input = document.createElement('paper-input');
        // input.setAttribute('always-float-label', 'true');
        // input.setAttribute('label', label);
        // input.setAttribute('value', '' + initialValue);
        // input.setAttribute('type', type);
        // if (type === 'number') {
        //     input.setAttribute('min', '' + min);
        //     input.setAttribute('max', '' + max);
        // }
        // input.className = 'param-input';
        this.paramContainer.appendChild(input);

        // Update the parent when this changes.
        input.addEventListener('input', (event) => {
            if (type === 'number') {
                // tslint:disable-next-line:no-any
                setValue((event.target).valueAsNumber);
            } else {
                // tslint:disable-next-line:no-any
                setValue((event.target).value);
            }
            this.ganPlayground.layerParamChanged();
        });
        setValue(initialValue);
    }
}


var dl = deeplearn;
var Array1D = dl.Array1D;
var Array2D = dl.Array2D;
var Array4D = dl.Array4D;
var conv_util = dl.conv_util;
var Graph = dl.Graph;
var Initializer = dl.Initializer;
var NDArrayInitializer = dl.NDArrayInitializer;
var Tensor = dl.Tensor;
var util = dl.util;
var VarianceScalingInitializer = dl.VarianceScalingInitializer;
var ZerosInitializer = dl.ZerosInitializer;



// export type LayerName = 'Fully connected' | 'ReLU' | 'Convolution' |
//     'Max pool' | 'Reshape' | 'Flatten';

/**
 * Creates a layer builder object.
 *
 * @param layerName The name of the layer to build.
 * @param layerBuilderJson An optional LayerBuilder JSON object. This doesn't
 *     have the prototype methods on them as it comes from serialization. This
 *     method creates the object with the necessary prototype methods.
 */
function getLayerBuilder(
    layerName, layerBuilderJson = null) {
    let layerBuilder;
    switch (layerName) {
        case 'Fully connected':
            layerBuilder = new FullyConnectedLayerBuilder();
            break;
        case 'ReLU':
            layerBuilder = new ReLULayerBuilder();
            break;
        case 'Convolution':
            layerBuilder = new Convolution2DLayerBuilder();
            break;
        case 'Max pool':
            layerBuilder = new MaxPoolLayerBuilder();
            break;
        case 'Reshape':
            layerBuilder = new ReshapeLayerBuilder();
            break;
        case 'Flatten':
            layerBuilder = new FlattenLayerBuilder();
            break;
        default:
            throw new Error('Layer builder for ' + layerName + ' not found.');
    }

    // For layer builders passed as serialized objects, we create the objects and
    // set the fields.
    if (layerBuilderJson != null) {
        for (const prop in layerBuilderJson) {
            if (layerBuilderJson.hasOwnProperty(prop)) {
                // tslint:disable-next-line:no-any
                (layerBuilder)[prop] = (layerBuilderJson)[prop];
            }
        }
    }
    return layerBuilder;
}

// export interface LayerParam {
//     label: string;
//     initialValue(inputShape: number[]): number | string;
//     type: 'number' | 'text';
//     min ? : number;
//     max ? : number;
//     setValue(value: number | string): void;
//     getValue(): number | string;
// }

// export type LayerWeightsDict = {
//     [name: string]: number[]
// };

// export interface LayerBuilder {
//     layerName: LayerName;
//     getLayerParams(): LayerParam[];
//     getOutputShape(inputShape: number[]): number[];
//     addLayer(
//         g: Graph, network: Tensor, inputShape: number[], index: number,
//         weights ? : LayerWeightsDict | null): Tensor;
//     addLayerMultiple(
//         g: Graph, networks: Tensor[], inputShape: number[], name: string,
//         weights ? : LayerWeightsDict | null): Tensor[];
//     // Return null if no errors, otherwise return an array of errors.
//     validate(inputShape: number[]): string[] | null;
// }

class FullyConnectedLayerBuilder {

    constructor() {
        this.layerName = 'Fully connected';
        this.hiddenUnits = null;
    }

    getLayerParams() {
        return [{
            label: 'Hidden units',
            initialValue: (inputShape) => 10,
            type: 'number',
            min: 1,
            max: 1000,
            setValue: (value) => this.hiddenUnits = value,
            getValue: () => this.hiddenUnits
        }];
    }

    getOutputShape(inputShape) {
        return [this.hiddenUnits];
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     const inputSize = util.sizeFromShape(inputShape);
    //     const wShape = [this.hiddenUnits, inputSize];

    //     let weightsInitializer;
    //     let biasInitializer;
    //     if (weights != null) {
    //         weightsInitializer =
    //             new NDArrayInitializer(Array2D.new(wShape, weights['W']));
    //         biasInitializer = new NDArrayInitializer(Array1D.new(weights['b']));
    //     } else {
    //         weightsInitializer = new VarianceScalingInitializer();
    //         biasInitializer = new ZerosInitializer();
    //     }

    //     const useBias = true;
    //     return g.layers.dense(
    //         'fc1', network, this.hiddenUnits, null, useBias, weightsInitializer,
    //         biasInitializer);
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {
        const inputSize = util.sizeFromShape(inputShape);
        const wShape = [inputSize, this.hiddenUnits];

        let w;
        let b;

        if (weights != null) {
            w = Array2D.new(wShape, weights[name + '-fc-w']);
            b = Array1D.new(weights[name + '-fc-b']);
        } else {
            w = Array2D.randTruncatedNormal(wShape, 0, 0.1);
            b = Array1D.zeros([this.hiddenUnits]);
        }
        const wTensor = g.variable(name + '-fc-w', w);
        const bTensor = g.variable(name + '-fc-b', b);

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(
                g.add(g.matmul(networks[i], wTensor), bTensor)
            );
        }
        return returnedTensors;
    }

    validate(inputShape) {
        if (inputShape.length !== 1) {
            return ['Input shape must be a Array1D.'];
        }
        return null;
    }
}

class ReLULayerBuilder {

    constructor() {
        this.layerName = 'ReLU';
    }

    getLayerParams() {
        return [];
    }

    getOutputShape(inputShape) {
        return inputShape;
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     return g.relu(network);
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(g.relu(networks[i]));
        }
        return returnedTensors;
    }

    validate(inputShape) {
        return null;
    }
}

class Convolution2DLayerBuilder {

    constructor() {
        this.layerName = 'Convolution';
        this.fieldSize = null;
        this.stride = null;
        this.zeroPad = null;
        this.outputDepth = null;
    }

    getLayerParams() {
        return [{
                label: 'Field size',
                initialValue: (inputShape) => 3,
                type: 'number',
                min: 1,
                max: 100,
                setValue: (value) => this.fieldSize = value,
                getValue: () => this.fieldSize
            },
            {
                label: 'Stride',
                initialValue: (inputShape) => 1,
                type: 'number',
                min: 1,
                max: 100,
                setValue: (value) => this.stride = value,
                getValue: () => this.stride
            },
            {
                label: 'Zero pad',
                initialValue: (inputShape) => 0,
                type: 'number',
                min: 0,
                max: 100,
                setValue: (value) => this.zeroPad = value,
                getValue: () => this.zeroPad
            },
            {
                label: 'Output depth',
                initialValue: (inputShape) =>
                    this.outputDepth != null ? this.outputDepth : 1,
                type: 'number',
                min: 1,
                max: 1000,
                setValue: (value) => this.outputDepth = value,
                getValue: () => this.outputDepth
            }
        ];
    }

    getOutputShape(inputShape) {
        return conv_util.computeOutputShape3D(
            inputShape, this.fieldSize,
            this.outputDepth, this.stride, this.zeroPad);
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     const wShape = [this.fieldSize, this.fieldSize, inputShape[2], this.outputDepth];
    //     let w;
    //     let b;
    //     if (weights != null) {
    //         w = Array4D.new(wShape, weights['W']);
    //         b = Array1D.new(weights['b']);
    //     } else {
    //         w = Array4D.randTruncatedNormal(wShape, 0, 0.1);
    //         b = Array1D.zeros([this.outputDepth]);
    //     }
    //     const wTensor = g.variable(`conv2d-${index}-w`, w);
    //     const bTensor = g.variable(`conv2d-${index}-b`, b);
    //     return g.conv2d(
    //         network, wTensor, bTensor, this.fieldSize, this.outputDepth,
    //         this.stride, this.zeroPad);
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {
        const wShape = [this.fieldSize, this.fieldSize, inputShape[2], this.outputDepth];
        let w;
        let b;
        if (weights != null) {
            w = Array4D.new(wShape, weights[name + '-conv2d-w']);
            b = Array1D.new(weights[name + '-conv2d-b']);
        } else {
            w = Array4D.randTruncatedNormal(wShape, 0, 0.1);
            b = Array1D.zeros([this.outputDepth]);
        }
        const wTensor = g.variable(name + '-conv2d-w', w);
        const bTensor = g.variable(name + '-conv2d-b', b);

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(
                g.conv2d(
                    networks[i], wTensor, bTensor, this.fieldSize,
                    this.outputDepth, this.stride, this.zeroPad)
            );
        }
        return returnedTensors;
    }

    validate(inputShape) {
        if (inputShape.length !== 3) {
            return ['Input shape must be a Array3D.'];
        }
        return null;
    }
}

class MaxPoolLayerBuilder {

    constructor() {

        this.layerName = 'Max pool';
        this.fieldSize = null;
        this.stride = null;
        this.zeroPad = null;
    }

    getLayerParams() {
        return [{
                label: 'Field size',
                initialValue: (inputShape) => 3,
                type: 'number',
                min: 1,
                max: 100,
                setValue: (value) => this.fieldSize = value,
                getValue: () => this.fieldSize
            },
            {
                label: 'Stride',
                initialValue: (inputShape) => 1,
                type: 'number',
                min: 1,
                max: 100,
                setValue: (value) => this.stride = value,
                getValue: () => this.stride
            },
            {
                label: 'Zero pad',
                initialValue: (inputShape) => 0,
                type: 'number',
                min: 0,
                max: 100,
                setValue: (value) => this.zeroPad = value,
                getValue: () => this.zeroPad
            }
        ];
    }

    getOutputShape(inputShape) {
        return conv_util.computeOutputShape3D(
            inputShape, this.fieldSize, inputShape[2],
            this.stride, this.zeroPad);
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     return g.maxPool(network, this.fieldSize, this.stride, this.zeroPad);
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(g.maxPool(networks[i], this.fieldSize,
                this.stride, this.zeroPad));
        }
        return returnedTensors;
    }

    validate(inputShape) {
        if (inputShape.length !== 3) {
            return ['Input shape must be a Array3D.'];
        }
        return null;
    }
}

class ReshapeLayerBuilder {


    constructor() {

        this.layerName = 'Reshape';
        this.outputShape = null;
    }
    getLayerParams() {
        return [{
            label: 'Shape (comma separated)',
            initialValue: (inputShape) => inputShape.join(', '),
            type: 'text',
            setValue: (value) => this.outputShape =
                value.split(',').map((value) => +value),
            getValue: () => this.outputShape.join(', ')
        }];
    }

    getOutputShape(inputShape) {
        return this.outputShape;
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     return g.reshape(network, this.outputShape);
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(g.reshape(networks[i], this.outputShape));
        }
        return returnedTensors;
    }

    validate(inputShape) {
        const inputSize = util.sizeFromShape(inputShape);
        const outputSize = util.sizeFromShape(this.outputShape);
        if (inputSize !== outputSize) {
            return [
                `Input size (${inputSize}) must match output size (${outputSize}).`
            ];
        }
        return null;
    }
}

class FlattenLayerBuilder {

    constructor() {
        this.layerName = 'Flatten';
    }

    getLayerParams() {
        return [];
    }

    getOutputShape(inputShape) {
        return [util.sizeFromShape(inputShape)];
    }

    // addLayer(
    //     g, network, inputShape, index,
    //     weights) {
    //     return g.reshape(network, this.getOutputShape(inputShape));
    // }

    addLayerMultiple(
        g, networks, inputShape, name,
        weights) {

        const returnedTensors = []
        for (let i = 0; i < networks.length; i++) {
            returnedTensors.push(g.reshape(networks[i],
                this.getOutputShape(inputShape)));
        }
        return returnedTensors;
    }

    validate(inputShape) {
        return null;
    }
}