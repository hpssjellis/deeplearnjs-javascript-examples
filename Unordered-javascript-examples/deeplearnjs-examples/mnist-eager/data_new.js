
const math = dl.ENV.math;
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
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }
    nextTrainBatch(batchSize) {
        return this.nextBatch(batchSize, this.trainingData, () => {
            this.shuffledTrainIndex =
                (this.shuffledTrainIndex + 1) % this.trainIndices.length;
            return this.trainIndices[this.shuffledTrainIndex];
        });
    }
    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, this.testData, () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }
    nextBatch(batchSize, data, index) {
        let xs = null;
        let labels = null;
        for (let i = 0; i < batchSize; i++) {
            const idx = index();
            const x = data[0][idx].reshape([1, 784]);
            xs = concatWithNulls(xs, x);
            const label = data[1][idx].reshape([1, 10]);
            labels = concatWithNulls(labels, label);
        }
        return { xs, labels };
    }
    async load() {
        this.dataset = new dl.XhrDataset(mnistConfig);
        await this.dataset.fetchData();
        this.dataset.normalizeWithinBounds(0, -1, 1);
        this.trainingData = this.getTrainingData();
        this.testData = this.getTestData();
        this.trainIndices =
            dl.util.createShuffledIndices(this.trainingData[0].length);
        this.testIndices = dl.util.createShuffledIndices(this.testData[0].length);
    }
    getTrainingData() {
        const [images, labels] = this.dataset.getData();
        const end = Math.floor(TRAIN_TEST_RATIO * images.length);
        return [images.slice(0, end), labels.slice(0, end)];
    }
    getTestData() {
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
function concatWithNulls(ndarray1, ndarray2) {
    if (ndarray1 == null && ndarray2 == null) {
        return null;
    }
    if (ndarray1 == null) {
        return ndarray2;
    }
    else if (ndarray2 === null) {
        return ndarray1;
    }
    return math.concat2D(ndarray1, ndarray2, 0);
}
