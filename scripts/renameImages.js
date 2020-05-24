const fs = require('fs');

function renameImages(symbolName, startNumber, oldPath, newPath) {
  let index = startNumber - 1;
  return fs.readdir(oldPath, (error, fileNames) => {
    return fileNames.forEach(fileName => {
      index++;
      return fs.rename(`${oldPath}/${fileName}`, `${newPath}/${symbolName}${index}.jpg`, (error) => {
        if (error) {
          throw error;
        }    
      })
    })
  })
    
}

// npm run renameImages [symbolName] [startNumber] [oldPath] [newPath]
renameImages(process.argv[2], process.argv[3], process.argv[4], process.argv[5]);