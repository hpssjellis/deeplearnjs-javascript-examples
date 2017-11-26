
 // To run as a webpage change extension form.ts to .html
 // The comment out the line below

//import * as dl from 'deeplearn';  
//import * as squeezenet from 'deeplearn-squeezenet';


async function myCheckImage(){
  dl = deeplearn
  const catImage = document.getElementById('cat');
  const math = new dl.NDArrayMathGPU();
  const squeezeNet = new squeezenet.SqueezeNet(math);
  await squeezeNet.load();

  const image = dl.Array3D.fromPixels(document.getElementById('cat'));
  const inferenceResult = await squeezeNet.predict(image);

  const topClassesToProbs = await squeezeNet.getTopKClasses(inferenceResult.logits, 10);

  document.getElementById('myDiv01').innerHTML = ''   // clear the div
  for (const className in topClassesToProbs) {
    
    document.getElementById('myDiv01').innerHTML += topClassesToProbs[className].toFixed(5) + ' : '+ className + '<br>'
  }
}