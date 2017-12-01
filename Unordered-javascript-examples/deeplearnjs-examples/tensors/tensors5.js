
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
  const my2D = dl.Array2D.new([3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  //const myOutput = await my2D.data()
  document.getElementById('myDiv01').innerHTML = 'Printing a 2D array shaped [3, 4] [rows, columns] <br><br>'
  document.getElementById('myDiv01').innerHTML += 'Original <br>'
  // print as a 2 x 3 table
  await myTensorTable('myDiv01', my2D, 4)  


  document.getElementById('myDiv01').innerHTML += 'Flattened <br>'
  my2D.flatten()
  myTensorTable('myDiv01', my2D, 4)  




 }
 
