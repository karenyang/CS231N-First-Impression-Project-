import os
const destFile = 'train-2/'
const extractPrefix = 'train80'

if __name__ == __main__: 
  for fileName in os.listdir(path):
    if fileName.startswith(extractPrefix):
      os.system('unzip ' + fileName)

  for fileName in os.listdir(path):
    newFileName = fileName
    while newFileName.startswith('-'):
      newFileName = newFileName[1:]
    os.system('mv ./' + fileName + ' ' + destFile + newFileName)
	
