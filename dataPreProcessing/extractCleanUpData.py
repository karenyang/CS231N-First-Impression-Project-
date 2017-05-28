import os

destFile = 'train-1-2-4/'
extractPrefix = 'training80'

def unzipAllFlsModNameMove():
  for fileName in os.listdir('./'):
   if fileName.startswith(extractPrefix):
     os.system('unzip ' + fileName)

  for fileName in os.listdir('./'):
    if(fileName.startswith('train-')):
    	continue
    newFileName = fileName
    while newFileName.startswith('-'):
      newFileName = newFileName[1:]
    os.system('mv ./' + fileName + ' ' + destFile + newFileName)


if __name__ == '__main__': 
  dirName = sys.argv[1]
  deleteMoviesWithFramesExtracted(dirName)

