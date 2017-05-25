import sys 
import random # for splitting into train/val
import os
import pickle # for resetting name 
from PIL import Image # for data validation 
 
def splitIntoTrainVal(directory, ratio_of_train):
  dirListing = os.listdir(directory)
  random.shuffle(dirListing)
  numTrain = int(len(dirListing) * ratio_of_train)
  
  trainFolder = os.path.join(directory, 'train')
  valFolder = os.path.join(directory,'val')
  os.system('mkdir ' +  trainFolder)
  os.system('mkdir ' +  valFolder)
  for f in dirListing[:numTrain]:
    print os.path.join(directory, f),os.path.join( trainFolder, f)
    os.rename(os.path.join(directory, f),os.path.join( trainFolder, f))
  for f in dirListing[numTrain:]:
    os.rename(os.path.join(directory, f),os.path.join( valFolder, f))

def resetNamesBack(dr):
  fileEnding ='_50uniform' #TODO: figure out how to make more general

  annotation_filename = dr + "/annotation_training.pkl"
  with open(annotation_filename, 'rb') as f:
        label_dicts = pickle.load(f) 

  for fileName in os.listdir(dr):
    fileName = fileName.replace(fileEnding,'.mp4')
    if(label_dicts.get(fileName)== None):
        print fileName, 'not in dictionary'
        fileNameCp  = fileName
    for i in range(3):
        fileNameCp  = '-' + fileNameCp
        if(label_dicts.get(fileNameCp) != None):
            print 'renaming', os.path.join(dr, fileName),os.path.join( dr, fileNameCp)
            os.rename(os.path.join(dr, fileName),os.path.join( dr, fileNameCp))

def validateAllData(path):
    numClean = 0
    numDirty = 0
    directories = [d for d in os.listdir('./') if os.path.isdir(d)]
    for dirName in directories:
        targetDir = os.path.join(path, dirName)
        for f in os.listdir(targetDir):
            fName = os.path.join(targetDir, f)
            try:
                im = Image.open(fName).convert('RGB')
                numClean = numClean + 1
            except ValueError:
                print "couldn't open " + fName
                numDirty = numDirty +1 
                print "numDirty" , numDirty, "numClean", numClean
                os.system('rm ' + dirty)
    print "overall ", numClean, "clean"
    
def extractFromZipMove():
  destFile = 'train-2/'
  extractPrefix = 'train80'
  for fileName in os.listdir(path):
    if fileName.startswith(extractPrefix):
      os.system('unzip ' + fileName)

  for fileName in os.listdir(path):
    newFileName = fileName
    while newFileName.startswith('-'):
      newFileName = newFileName[1:]
    os.system('mv ./' + fileName + ' ' + destFile + newFileName)


if __name__ == '__main__': 
    sourceFile = sys.argv[1]
    # resetNamesBack(sourceFile)
    validateAllData(sourceFile)
