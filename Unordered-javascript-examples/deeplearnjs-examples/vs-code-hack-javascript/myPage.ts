<script src="https://unpkg.com/deeplearn@0.3.11/dist/deeplearn.js"> </script> 
<script>  dl = deeplearn </script>


<script>

import * as dl from 'deeplearn';
async function runExample() {
    const math = new dl.NDArrayMathGPU();
    const a = dl.Array1D.new([1, 2, 3]);
    const b = dl.Scalar.new(2);
    const result = math.add(a, b);
    
     document.getElementById('myDiv01').innerHTML = await result.data() + 'aaa<br>'
}

</script>

<input type=button value="wow" onclick="{
    runExample()
  }">
  
  <div id="myDiv01">...</div>
  
