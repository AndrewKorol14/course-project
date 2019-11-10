const fileExplorer = document.getElementById('fileExplorer');
fileExplorer.addEventListener("change", outputImage);
const trainButton = document.getElementById('train');
import {run} from './app.js';
fileExplorer.addEventListener("click", run);

function outputImage(){
  let img = document.createElement("img");
  let file = this.files[0];
  img.classList.add("obj");
  img.file = file;
  document.getElementById('image').appendChild(img); 
  img.style.width = "500px";
  img.style.height = "500px";
  img.style.marginTop = "20px";
  let reader = new FileReader();
  reader.onload = (function(aImg) { 
    return function(e) { 
      aImg.src = e.target.result;
      let button = document.createElement("button");
      button.textContent = "Прогнозировать";
      button.style.marginLeft = "20px";
      button.style.position = "absolute";
      button.style.top = "50%";
      button.style.transform = "translate(0, -50%)";
      document.getElementById('image').appendChild(button);
    }; 
  })(img);
  reader.readAsDataURL(file);
}