//import * as dl from 'deeplearn';

//const math = dl.ENV.math;

const TRAIN_TEST_RATIO = 5 / 6;

const mnistConfig = {
  'data': [
    {
      'name': 'images',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_images.png',
      'dataType': 'png',
      'shape': [28, 28, 1]
    },
    {
      'name': 'labels',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_labels_uint8',
      'dataType': 'uint8',
      'shape': [10]
    }
  ],
  modelConfigs: {}
};

class MnistData {

  constructor(name) {
     this.shuffledTrainIndex = 0;
     this.shuffledTestIndex = 0;
  }


  async nextTrainBatch(batchSize) {
        var _this = this;
        return this.nextBatch(batchSize, this.trainingData, function () {
            _this.shuffledTrainIndex =
                (_this.shuffledTrainIndex + 1) % _this.trainIndices.length;
            return _this.trainIndices[_this.shuffledTrainIndex];
        });
  }

  async nextTestBatch(batchSize) {
        var _this = this;
        return this.nextBatch(batchSize, this.testData, function () {
            _this.shuffledTestIndex =
                (_this.shuffledTestIndex + 1) % _this.testIndices.length;
            return _this.testIndices[_this.shuffledTestIndex];
        });
  }

  async nextBatch(batchSize, data, index) {
        var xs = null;
        var labels = null;
        for (var i = 0; i < batchSize; i++) {
            var idx = index();
            var x = data[0][idx].reshape([1, 784]);
            xs = concatWithNulls(xs, x);
            var label = data[1][idx].reshape([1, 10]);
            labels = concatWithNulls(labels, label);
        }
        return { xs, labels};
  }

  async load() {
    this.dataset = new dl.XhrDataset(mnistConfig);
    console.log('1')
    await this.dataset.fetchData();
    console.log('2')
    this.dataset.normalizeWithinBounds(0, -1, 1);
    console.log('3')
    this.trainingData = this.getTrainingData();
    console.log('this.trainingData[0].length'+this.trainingData[1].length)
    this.testData = this.getTestData();
    console.log('5')

    this.trainIndices = dl.util.createShuffledIndices(this.trainingData[0].length);
     console.log('6')
    this.testIndices = dl.util.createShuffledIndices(this.testData[0].length);
      console.log('7')
    
  }

  async getTrainingData() {
        console.log('3A')
    const [images, labels] = this.dataset.getData();
    
        console.log('3B')
    const end = Math.floor(TRAIN_TEST_RATIO * images.length);
    
        console.log('3C')
    return [images.slice(0, end), labels.slice(0, end)];
  }

  async getTestData(){
    const data = this.dataset.getData();
    if (data == null) {
      return null;
    }
    const [images, labels] = this.dataset.getData();
    const start = Math.floor(TRAIN_TEST_RATIO * images.length);
    return [images.slice(start), labels.slice(start)];
  }
  
  
}

/**
 * TODO(nsthorat): Add math.stack, similar to np.stack, which will avoid the
 * need for us allowing concating with null values.
 */
function concatWithNulls(ndarray1,ndarray2){
  if (ndarray1 == null && ndarray2 == null) {
    return null;
  }
  if (ndarray1 == null) {
    return ndarray2;
  } else if (ndarray2 === null) {
    return ndarray1;
  }
  return math.concat2D(ndarray1, ndarray2, 0);
}
