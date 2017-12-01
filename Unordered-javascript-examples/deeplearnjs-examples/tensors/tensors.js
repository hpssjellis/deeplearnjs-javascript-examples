// To run as a webpage 
// The comment out the import lines below. 
// To use IntelliSense uncomment the import lines

//import * as dl from 'deeplearn';  

// div as a string, Tensor as a tensor, NumberRows as an integer, Title as a string
async function myTensorTable(myDiv, myOutTensor, myCols, myTitle){   

  
 document.getElementById(myDiv).innerHTML += myTitle + '<br>'

 const myOutput = await myOutTensor.data()
 myTemp = '<table border=3><tr>'
   for (myCount = 0;    myCount <= myOutTensor.size - 1;   myCount++){   
     myTemp += '<td>'+ myOutput[myCount] + '</td>'
     if (myCount % myCols == myCols-1){
         myTemp += '</tr><tr>'
     }
   }   
   myTemp += '</tr></table>'
   document.getElementById(myDiv).innerHTML += myTemp + '<br>'
}


async function myTest01(){
  dl = deeplearn 
  const matrixShape = [2, 3];  // 2 rows, 3 columns.
  const myWeights  = dl.Array2D.new([3, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
  const myInputs = dl.Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const myBias    = dl.Scalar.new(2.3);

 //const myOutput = await my2D.data()
 await myTensorTable('myDiv01', myWeights, 3, 'Weights [3x3]')  


 await myTensorTable('myDiv01', myInputs, 3, 'Inputs [3x3]')  

  
  var myFlat = myInputs.flatten(myInputs)  
  await myTensorTable('myDiv01', myFlat, 1, 'flatten the 2D Inputs array into a 1D array')    
  

 const myBiasOutput = await myBias.data();
 document.getElementById('myDiv01').innerHTML += 'Bias Scalar = ' + myBiasOutput + ' <br><hr>'


 //Generic math ability set
 var myAlgorithm = new dl.NDArrayMathGPU()


  var  myProduct = myAlgorithm.matMul(myWeights, myInputs)
  await myTensorTable('myDiv01', myProduct, 3, 'Product of Weights x Inputs')  


 const myMatrix1  = dl.Array2D.new([2, 2], [3, 2, 5, 4]);
 const myMatrix2 =  dl.Array2D.new([2, 2], [2, 3, 4, 5]);

 await myTensorTable('myDiv01', myMatrix1, 2, 'Table A')  
 await myTensorTable('myDiv01', myMatrix2, 2, 'Table B')  



  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2)
  document.getElementById('myDiv01').innerHTML += ' A * B Matmul 2D = ' + myProduct.getValues()  +'<br>'   

  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2, 'TRANSPOSED', 'REGULAR')
  document.getElementById('myDiv01').innerHTML += 'A^T * B Matmul 2D = ' + myProduct.getValues()  +'<br>'  

  var  myProduct = myAlgorithm.matMul(myMatrix2, myMatrix1)
  document.getElementById('myDiv01').innerHTML += 'B * A Matmul 2D = ' + myProduct.getValues()  +'<br>' 

  var  myProduct = myAlgorithm.matMul(myMatrix2, myMatrix1, 'REGULAR', 'TRANSPOSED')
  document.getElementById('myDiv01').innerHTML += ' B * A^T  Matmul 2D = ' + myProduct.getValues()  +'<br><hr>'   


  


 const w = new dl.Array1D.new([0.7, -0.3])
 const i = new dl.Array1D.new([0.1, 0.8])
 const b = dl.Scalar.new(2.1)

 await myTensorTable('myDiv01', w, 2, 'w as weights')  
 await myTensorTable('myDiv01', i, 2, 'i as inputs ')  

 document.getElementById('myDiv01').innerHTML += 'b as Bias= '+ b.getValues()  +'<br><Br>'
 
 myAlgorithm.scope(function() {
   var myDot = myAlgorithm.dotProduct(w, i)
   var mySum = myAlgorithm.add(myDot, b)
   var mySig = myAlgorithm.sigmoid(mySum)
   document.getElementById('myDiv01').innerHTML += 'sigmoid(add( dot(w, i)+ b) ) = '+mySig.getValues()  +'<br>'
 })

 myAlgorithm.scope(function() {
    var myDot = myAlgorithm.dotProduct(w, i)
    var mySum = myAlgorithm.add(myDot, b)
    var mySig = myAlgorithm.sinh(mySum)
    document.getElementById('myDiv01').innerHTML += 'sinH(add( dot(w, i)+ b) ) = '+mySig.getValues()  +'<br>'
 })


 myAlgorithm.scope(function() {
    var myDot = myAlgorithm.dotProduct(w, i)
    var mySum = myAlgorithm.add(myDot, b)
    var mySig = myAlgorithm.relu(mySum)
    document.getElementById('myDiv01').innerHTML += 'relu(add( dot(w, i)+ b) ) = '+mySig.getValues()  +'<br><hr>'
 })
    


  
  
  
}
