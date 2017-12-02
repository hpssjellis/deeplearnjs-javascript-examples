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
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Array3D = dl.Array3D;
var Array4D = dl.Array4D;
var Array1D = dl.Array1D;
var ENV = dl.ENV;

function computeWeightsShape4D(
    inputDepth, outputDepth, filterHeight,
    filterWidth) {
    return [filterHeight, filterWidth, inputDepth, outputDepth];
}

const canvas = document.getElementById(`plot`);
var chart = create_chart(canvas);

chartData = [{
        label: 'deeplearnjs -- t1',
        backgroundColor: window.chartColors.red,
        borderColor: window.chartColors.red,
        data: [],
        fill: false,
        pointRadius: 0,
        pointHitRadius: 5,
        borderWidth: 1,
        lineTension: 0,
    },
    {
        label: "convnetjs -- t2",
        backgroundColor: window.chartColors.blue,
        borderColor: window.chartColors.blue,
        data: [],
        fill: false,
        pointRadius: 0,
        pointHitRadius: 5,
        borderWidth: 1,
        lineTension: 0,
    }
]
config.data.datasets = chartData;
chart.update();


var table = document.getElementById(`divTable`);
init_table(table, ['Size', 't1 (ms)', 't2 (ms)']);

class ConvBenchmark {

    constructor(libName) {
        this.libName = libName
        this.btn = document.getElementById('buttontp' + "_" + this.libName);
        this.paused = true;
        this.btn.addEventListener('click', (event) => {
            this.toggle_pause();
            ga('send', 'event', 'deeplearn_conv_benchmark', 'click', `Run Benchmark ${this.libName}`, this.libName === 'dljs' ? 30 : 31);
        });
    }

    toggle_pause() {
        this.paused = !this.paused;
        if (this.paused) {
            this.btn.value = 'Run Benchmark'
            this.btn.disabled = false
        } else {
            this.btn.value = 'Running...';
            this.btn.disabled = true
        }
    }

    run(size, opType, params) {

        if (this.libName === 'dljs') {
            var math = new NDArrayMathGPU();
            var gpgpu = math.getGPGPUContext();
        }

        const inDepth = params.inDepth;
        const inShape = [size, size, inDepth];
        const outDepth = params.outDepth;
        const filterSize = params.filterSize;
        const stride = params.stride;
        let zeroPad = params.zeroPad;

        // outputRows>=0 needs to be asserted
        var outputRows = (inShape[0] - filterSize + 2 * zeroPad) / stride + 1;

        function isInt(a) {
            return a % 1 === 0;
        }

        if (outputRows <= 0) {
            console.log(`input size ${size} doesn't satisfy assertion outputRows>0, given inputRows=${size}, filterSize=${filterSize}, zeroPad=${zeroPad}, stride=${stride}, minimal size needs to be ${filterSize + 2 * zeroPad - stride}`)
        } else if (!isInt(outputRows)) {
            console.log(`outputRows ${outputRows} is not int`)
        }

        function find_min_pad(i, f, s) {

            for (let n = 0;; n++) {
                var z = (n * s - i + f) / 2;
                if (z > 0) {
                    console.log('found suitable z=' + z)
                    return z
                }
            }

        }
        zeroPad = find_min_pad(size, filterSize, stride);
        console.log(`min pad ${zeroPad} applied`)

        let benchmark;
        let out;
        let b;
        let x;

        if (this.libName === 'dljs') { //deeplearnjs

            if (opType === 'regular') {
                x = Array3D.randUniform(inShape, -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, params.outDepth, filterSize, filterSize);
                var W = Array4D.randUniform(wShape, -1, 1);
                // const b = Array1D.randUniform([outDepth], -1, 1);

                benchmark = () => {
                    out = math.conv2d(x, W, null, stride, zeroPad); //bias=null, pad =0, this padding will be applied on four borders of input : left right top bottom
                }


            } else if (opType === 'transposed') {

                x = Array3D.randUniform([size, size, regParams.outDepth], -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, params.outDepth, filterSize, filterSize);
                W = Array4D.randUniform(wShape, -1, 1);

                // no bias for conv transposed
                benchmark = () => {
                    out = math.conv2dTranspose(x, W, [size, size, inDepth], stride, pad);
                };


            } else if (opType === 'depthwise') {
                x = Array3D.randUniform(inShape, -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, params.channelMul, filterSize, filterSize);
                W = Array4D.randUniform(wShape, -1, 1);

                // no bias for depth wise conv
                benchmark = () => {
                    out = math.depthwiseConv2D(x, W, stride, pad);
                };

            } else {
                throw new Error(`Unknown option ${opType}`);
            }

        } else { // convnetjs

            console.assert(opType === 'regular', `unsupported conv op type`)

            x = new convnetjs.Vol(size, size, inDepth); // 128,128,3 
            const opt = {
                in_sx: size,
                in_sy: size,
                in_depth: inDepth,
                sx: filterSize,
                filters: outDepth,
                stride: stride,
                pad: zeroPad // this will be applied on four borders of input :left right top bottom
            }; // no bias, pad=0
            var layer = new convnetjs.ConvLayer(opt);

            benchmark = () => {
                out = layer.forward(x);
            }

        }

        const cleanup = () => {

            if (this.libName === 'dljs') {
                x.dispose();
                W.dispose();
                out.dispose();
                if (b != null) {
                    b.dispose();
                }
            } else {
                x = null;
                out = null;
            }

        };

        if (this.libName === 'dljs') {
            // Warmup.
            gpgpu.runQuery(benchmark);
            out.dispose();
        }

        let totalTime;

        if (this.libName === 'dljs') {

            if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
                totalTime = gpgpu.runQuery(benchmark);
            } else {
                const start = performance.now();

                benchmark();
                out.dataSync();

                totalTime = performance.now() - start;
            }
        } else {

            const start = performance.now();

            benchmark();

            totalTime = performance.now() - start;
        }

        console.log(`${this.libName} convolution output: ${out}`)
        cleanup();
        return totalTime;
    }

    run_test(params) {

        this.index = this.libName === 'dljs' ? 0 : 1;

        var bmrun = this;

        const runPromises = [];

        var sizes = [];
        for (let size = 16; size < 4000; size = size * 2) {
            var t = bmrun.run(size, 'regular', params)
            // t.then(data => console.log('conv on size', size, 'time', data, 'us'))
            runPromises.push(t);
            sizes.push(size)
        }

        if (config.data.labels.length !== sizes.length) {
            init_chart(config, sizes);
        }

        if (table.rows !== sizes.length + 1) {
            update_table_col(table, 0, sizes);
        }


        Promise.all(runPromises).then(results => {
            for (let i = 0; i < results.length; i++) {

                let resultString;
                let logString;
                let time = 0;
                let success = true;
                let size = sizes[i]
                try {
                    time = results[i].toFixed(3);
                    resultString = time + ' ms';
                    logString = resultString;
                } catch (e) {
                    success = false;
                    resultString = 'Error';
                    logString = e.message;
                }

                if (time >= 0) {
                    if (success) {
                        chartData[bmrun.index].data.push(
                            // {
                            // x: size,
                            // y: time
                            // }
                            time
                        );

                    }
                    // rowValues.push(resultString);
                }
                console.log(`[${size}]: ${logString}`);

            }

            var resultStringArray = [];
            chartData[bmrun.index].data.forEach((time) => {
                resultStringArray.push(time);
            })

            update_table_col(table, bmrun.index + 1, resultStringArray); // col0 is the size labels
            config.data.datasets = chartData;

            chart.update();

        });

        bmrun.toggle_pause();

    }
}

function init_chart(config, sizes) {

    config.data.labels = sizes;

    chart.update();

}

// (inputRows - fieldSize + 2 * zeroPad) / stride + 1) >=0 needs to be asserted
const convParams = {
    inDepth: 8,
    outDepth: 3,
    filterSize: 7,
    stride: 2,
    zeroPad: 0 // adjust zeroPad so that (inputSize - filterSize + 2* zeroPad) is integer
};

var bm_dljs = new ConvBenchmark('dljs');
var bm_cnjs = new ConvBenchmark('cnjs');

function run() {

    if (bm_dljs.paused === false) {

        setTimeout(function () {
            bm_dljs.run_test(convParams);
            run()
        }, 0);

    } else if (bm_cnjs.paused == false) {

        setTimeout(function () {
            bm_cnjs.run_test(convParams);
            run()
        }, 0);

    } else {
        setTimeout(function () {
            // console.log('outside')
            run();
        }, 1000);
    }

}


function start() {

    supported = detect_support();

    if (supported) {
        console.log('device & webgl supported')
        document.getElementById("buttontp_dljs").disabled = false;
    } else {
        console.log('device/webgl not supported')
        document.getElementById("buttontp_dljs").disabled = true;
    }

    run();
}