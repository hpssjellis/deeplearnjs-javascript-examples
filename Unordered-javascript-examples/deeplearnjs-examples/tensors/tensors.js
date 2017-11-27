
 // To run as a webpage 
 // The comment out the import lines below. 
 // To use IntelliSense uncomment the import lines

 //import * as dl from 'deeplearn';  


 
async function myTest01(){
  dl = deeplearn 
  const matrixShape = [2, 3];  // 2 rows, 3 columns.
  const my2D = dl.Array2D.new([2, 3], [10, 20, 30, 40, 50, 60]);
  const myOutput = await my2D.data()
  document.getElementById('myDiv01').innerHTML = 'Array2D values are '+ myOutput +'<br>'
  
  // print as a 2x3 table
  myTemp = '<table border=3><tr>'

  for (myCount = 0;    myCount <= my2D.size - 1;   myCount++){   
    myTemp += '<td>'+ myOutput[myCount] + '</td>'
    if (myCount % 3 == 2){
        myTemp += '</tr><tr>'
    }
  }
     
  myTemp += '</tr></table>'
  document.getElementById('myDiv01').innerHTML += myTemp + '<br>'
  



  

   
   
 }
 