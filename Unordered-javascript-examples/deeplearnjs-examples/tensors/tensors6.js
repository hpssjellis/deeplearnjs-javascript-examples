// To run as a webpage 
// The comment out the import lines below. 
// To use IntelliSense uncomment the import lines

//import * as dl from 'deeplearn';  


async function myTensorTable(myDiv, myOutTensor, myCols){   // div only as a string
 const myOutput = await myOutTensor.data()
 myTemp = '<br><table border=3><tr>'
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
  const myWeights  = dl.Array2D.new([3, 4], [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
  const myInputs = dl.Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const myBias    = dl.Scalar.new(2.3);

 //const myOutput = await my2D.data()
 document.getElementById('myDiv01').innerHTML = '<br>Printing myWeights [3, 4] [rows, columns] <br>'
 await myTensorTable('myDiv01', myWeights, 4)  


 document.getElementById('myDiv01').innerHTML += 'Printing myInputs [3, 4] [rows, columns] <br>'
 await myTensorTable('myDiv01', myInputs, 4)  


 const myBiasOutput = await myBias.data();
 document.getElementById('myDiv01').innerHTML += 'Printing myBias Scalar = ' + myBiasOutput + ' <br><hr>'






 //Generic math ability set
 var myAlgorithm = new dl.NDArrayMathGPU()






 const myMatrix1  = dl.Array2D.new([2, 2], [3, 2, 5, 4]);
 const myMatrix2 =  dl.Array2D.new([2, 2], [2, 3, 4, 5]);

 await myTensorTable('myDiv01', myMatrix1, 2)  
 await myTensorTable('myDiv01', myMatrix2, 2)  


 myAlgorithm.scope(function() {
  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2)
  document.getElementById('myDiv01').innerHTML += 'Matmul 2D = ' + myProduct.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])

  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2, 'TRANSPOSED', 'REGULAR')
  document.getElementById('myDiv01').innerHTML += 'A^T * B Matmul 2D = ' + myProduct.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])


  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2, 'TRANSPOSED', 'TRANSPOSED')
  document.getElementById('myDiv01').innerHTML += 'A^T * B^T Matmul 2D = ' + myProduct.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])



  var  myProduct = myAlgorithm.matMul(myMatrix1, myMatrix2, 'REGULAR', 'TRANSPOSED')
  document.getElementById('myDiv01').innerHTML += ' A * B^T  Matmul 2D = ' + myProduct.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])






})
  










 const w = new dl.Array1D.new([0.7, -0.3])

 const i = new dl.Array1D.new([0.1, 0.8])
 const b = dl.Scalar.new(2.1)


 
 await myTensorTable('myDiv01', w, 2)  
 await myTensorTable('myDiv01', i, 2)  

 myAlgorithm.scope(function() {
   var myDot = myAlgorithm.dotProduct(w, i)
   var mySum = myAlgorithm.add(myDot, b)
   var mySig = myAlgorithm.sigmoid(mySum)
   document.getElementById('myDiv01').innerHTML += 'easy sigmoid = '+mySig.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])
 })

 myAlgorithm.scope(function() {
    var myDot = myAlgorithm.dotProduct(w, i)
    var mySum = myAlgorithm.add(myDot, b)
    var mySig = myAlgorithm.sinh(mySum)
    document.getElementById('myDiv01').innerHTML += 'easy sinH = '+mySig.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])
 })


 myAlgorithm.scope(function() {
    var myDot = myAlgorithm.dotProduct(w, i)
    var mySum = myAlgorithm.add(myDot, b)
    var mySig = myAlgorithm.relu(mySum)
    document.getElementById('myDiv01').innerHTML += 'easy relu = '+mySig.getValues()  +'<br>...<br>'   // Float32Array([3, 4, 5])
 })
    



 








 

}
