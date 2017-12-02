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
var math = new dl.NDArrayMathGPU();
var a = dl.Array1D.new([1, 2, 3]);
var b = dl.Scalar.new(2);

var result = math.add(a, b);

// for(var propertyName in result) {console.log(typeof propertyName, propertyName)}

// // Option 1: With a Promise.
// result.getValuesAsync().then(data => console.log(data)); // Float32Array([3, 4, 5])

// // Option 2: Synchronous download of data. This is simpler, but blocks the UI.
// console.log(result.getValues());


// Option 1: With a Promise.
result.data().then(data => console.log(data));

// Option 2: Synchronous download of data. This is simpler, but blocks the UI.
console.log(result.dataSync());