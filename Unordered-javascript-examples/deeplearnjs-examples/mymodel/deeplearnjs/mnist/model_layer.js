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

class ModelLayer {

    constructor() {
        // super(); // always call super() first in the ctor.
        // this.addEventListener('click', e => this.drawRipple(e.offsetX, e.offsetY));
        this.inputShapeDisplay = null;
        this.outputShapeDisplay = null; //: string;
        this.layerNames = null; //: LayerName[];
        this.selectedLayerName = null; //: LayerName;
        this.hasError = null; //: boolean;
        this.errorMessages = null; //: string[];

        // this.modelBuilder = null; //: ModelBuilder;
        this.layerBuilder = null; // LayerBuilder;
        this.inputShape = null; //: number[];
        this.outputShape = null; //: number[];

        this.paramContainer = null; //: HTMLDivElement;
    }

    initialize(modelBuilder, inputShape) {
        // this.modelBuilder = modelBuilder; // document.currentScript
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
        //     modelBuilder.removeLayer(this);
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

    addLayer(
        g, network, index,
        weights) {
        return this.layerBuilder.addLayer(
            g, network, this.inputShape, index, weights);
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
        this.paramContainer.innerHTML = ' ';

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
        layerParamChanged();
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
        const input = document.createElement('div');
        // input.setAttribute('always-float-label', 'true');
        // input.setAttribute('label', label);
        // input.setAttribute('value', initialValue.toString());
        // input.setAttribute('type', type);
        // if (type === 'number') {
        //     input.setAttribute('min', min.toString());
        //     input.setAttribute('max', max.toString());
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
            layerParamChanged();
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

/**
 * Classes that specify operation parameters, how they affect output shape,
 * and methods for building the operations themselves. Any new ops to be added
 * to the model builder UI should be added here.
 */

// type LayerName =
//     'Fully connected' | 'ReLU' | 'Convolution' | 'Max pool' | 'Reshape' | 'Flatten';

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
            throw new Error(`Layer builder for ${layerName} not found.`);
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

// interface LayerParam {
//     label: string;
//     initialValue(inputShape: number[]): number | string;
//     type: 'number' | 'text';
//     min ? : number;
//     max ? : number;
//     setValue(value: number | string): void;
//     getValue(): number | string;
// }

// type LayerWeightsDict = {
//     [name: string]: number[]
// };

// interface LayerBuilder {
//     layerName: LayerName;
//     getLayerParams(): LayerParam[];
//     getOutputShape(inputShape: number[]): number[];
//     addLayer(
//         g: Graph, network: Tensor, inputShape: number[], index: number,
//         weights ? : LayerWeightsDict | null): Tensor;
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        const inputSize = util.sizeFromShape(inputShape);
        const wShape = [this.hiddenUnits, inputSize];

        let weightsInitializer;
        let biasInitializer;
        if (weights != null) {
            weightsInitializer =
                new NDArrayInitializer(Array2D.new(wShape, weights['W']));
            biasInitializer = new NDArrayInitializer(Array1D.new(weights['b']));
        } else {
            weightsInitializer = new VarianceScalingInitializer();
            biasInitializer = new ZerosInitializer();
        }

        const useBias = true;
        return g.layers.dense(
            'fc1', network, this.hiddenUnits, null, useBias, weightsInitializer,
            biasInitializer);
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        return g.relu(network);
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        const wShape = [this.fieldSize, this.fieldSize, inputShape[2], this.outputDepth];
        let w;
        let b;
        if (weights != null) {
            w = Array4D.new(wShape, weights['W']);
            b = Array1D.new(weights['b']);
        } else {
            w = Array4D.randTruncatedNormal(wShape, 0, 0.1);
            b = Array1D.zeros([this.outputDepth]);
        }
        const wTensor = g.variable(`conv2d-${index}-w`, w);
        const bTensor = g.variable(`conv2d-${index}-b`, b);
        return g.conv2d(
            network, wTensor, bTensor, this.fieldSize, this.outputDepth,
            this.stride, this.zeroPad);
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        return g.maxPool(network, this.fieldSize, this.stride, this.zeroPad);
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        return g.reshape(network, this.outputShape);
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

    addLayer(
        g, network, inputShape, index,
        weights) {
        return g.reshape(network, this.getOutputShape(inputShape));
    }

    validate(inputShape) {
        return null;
    }
}