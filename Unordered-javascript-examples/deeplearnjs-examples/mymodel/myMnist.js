

function myLoadDefaults(){

}


function myLoadMnist(){
  var myIncomingData = new deeplearn.InMemoryDataset()
  var myDatasetName = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png'
    myIncomingData.selectedDatasetName = myDatasetName;
    myIncomingData.selectedModelName = '';
    myIncomingData.dataSet = myIncomingData.dataSets[myDatasetName];
    myIncomingData.datasetDownloaded = false;
    myIncomingData..showDatasetStats = false;

    myIncomingData.dataSet.fetchData().then(() => {
      myIncomingData.datasetDownloaded = true;
      myIncomingData.applyNormalization(myIncomingData.selectedNormalizationOption);
      myIncomingData.setupDatasetStats();
      if (myIncomingData.isValid) {
        myIncomingData.createModel();
      }
      // Get prebuilt models.
      //this.populateModelDropdown();
    });

    myIncomingData.inputShape = myIncomingData.dataSet.getDataShape(0);
    myIncomingData.labelShape = myIncomingData.dataSet.getDataShape(1);
}


function myTrainMnist()  {
  alert('This trains a set number of images')
}



function myInfer() {
  alert('This infers every second a testing image and shows result percentages. No graphics needed')
}




