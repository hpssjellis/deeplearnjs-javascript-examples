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

class NDArrayLogitsVisualizer {
    // private logitLabelElements: HTMLElement[];
    // private logitVizElements: HTMLElement[];
    // private width: number;

    constructor(elt, _topk) {
        this.elt = elt;
        this.logitLabelElements = null;
        this.logitVizElements = null;
        this.width = null;

        this.TOP_K = _topk;

    }

    initialize(width, height) {
        this.width = width;
        this.logitLabelElements = [];
        this.logitVizElements = [];
        const container = document.createElement('div'); // this.querySelector('.logits-container');
        container.style.height = `${height}px`;
        // container.style.paddingLeft = '10px';

        for (let i = 0; i < this.TOP_K; i++) {
            const logitContainer = document.createElement('div');
            logitContainer.style.height = `${height / (this.TOP_K + 1)}px`;
            logitContainer.style.margin =
                `${height / ((2 * this.TOP_K) * (this.TOP_K + 1))}px 0`;
            logitContainer.className =
                'single-logit-container ndarray-logits-visualizer';

            const logitLabelElement = document.createElement('div');
            logitLabelElement.className = 'logit-label ndarray-logits-visualizer';
            this.logitLabelElements.push(logitLabelElement);

            const logitVizOuterElement = document.createElement('div');
            logitVizOuterElement.className =
                'logit-viz-outer ndarray-logits-visualizer';

            const logitVisInnerElement = document.createElement('div');
            logitVisInnerElement.className =
                'logit-viz-inner ndarray-logits-visualizer';
            logitVisInnerElement.innerHTML = '&nbsp;';
            logitVizOuterElement.appendChild(logitVisInnerElement);

            this.logitVizElements.push(logitVisInnerElement);

            logitContainer.appendChild(logitLabelElement);
            logitContainer.appendChild(logitVizOuterElement);
            container.appendChild(logitContainer);
        }

        this.elt.appendChild(container)
    }

    drawLogits(
        predictedLogits, labelLogits,
        labelClassNames = null) {
        const mathCpu = new NDArrayMathCPU();
        const labelClass = mathCpu.argMax(labelLogits).get();

        const topk = mathCpu.topK(predictedLogits, this.TOP_K);
        const topkIndices = topk.indices.getValues();
        const topkValues = topk.values.getValues();

        for (let i = 0; i < topkIndices.length; i++) {
            const index = topkIndices[i];
            this.logitLabelElements[i].innerText =
                labelClassNames ? labelClassNames[index] : index.toString();
            this.logitLabelElements[i].style.width =
                labelClassNames != null ? '100px' : '20px';
            this.logitVizElements[i].style.backgroundColor = index === labelClass ?
                'rgba(120, 185, 50, .84)' :
                'rgba(220, 10, 10, 0.84)';
            this.logitVizElements[i].style.width =
                `${Math.floor(100 * topkValues[i])}%`;
            this.logitVizElements[i].innerText =
                `${(100 * topkValues[i]).toFixed(1)}%`;
        }
    }
}






class NDArrayImageVisualizer {

    constructor(elt) {
        this.elt = elt;

        this.imageData = null;

        this.canvas = document.createElement('canvas'); // this.querySelector('#canvas');
        this.canvas.style.display = "table-cell";
        this.canvas.width = 0;
        this.canvas.height = 0;
        this.canvasContext =
            this.canvas.getContext('2d');
        this.canvas.style.display = 'none';
        this.elt.appendChild(this.canvas);
    }

    setShape(shape) {
        this.canvas.width = shape[1];
        this.canvas.height = shape[0];
    }

    setSize(width, height) {
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
    }

    saveImageDataFromNDArray(ndarray) {
        this.imageData = this.canvasContext.createImageData(
            this.canvas.width, this.canvas.height);
        if (ndarray.shape[2] === 1) {
            this.drawGrayscaleImageData(ndarray);
        } else if (ndarray.shape[2] === 3) {
            this.drawRGBImageData(ndarray);
        }
    }

    drawRGBImageData(ndarray) {
        let pixelOffset = 0;
        for (let i = 0; i < ndarray.shape[0]; i++) {
            for (let j = 0; j < ndarray.shape[1]; j++) {
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 1);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 2);
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    }

    drawGrayscaleImageData(ndarray) {
        let pixelOffset = 0;
        for (let i = 0; i < ndarray.shape[0]; i++) {
            for (let j = 0; j < ndarray.shape[1]; j++) {
                const value = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    }

    draw() {
        this.canvas.style.display = '';
        this.canvasContext.putImageData(this.imageData, 0, 0);
    }
}

