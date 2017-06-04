'''
Format of annotation_training.pkl
{
extraversion: {
  videoName1: value1,
  videoName2: value2,
  etc...
},
neuroticism:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
agreeableness:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
conscientiousness:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
interview:{
  videoName1: value1,
  videoName2: value2,
  etc...
},
openness:{
  videoName1: value1,
  videoName2: value2,
  etc...
}
}
'''

import pickle
import numpy as np
# create a small benchmark to see accuraccy results for each score based on random guessing

annotationsDictPath = '/home/noa_glaser/dataBig/annotation_training.pkl'

a = pickle.load(open(annotationsDictPath))

trueVal = np.array([
  a['extraversion'].values(),
  a['neuroticism'].values(),
  a['agreeableness'].values(),
  a['conscientiousness'].values(),
  a['openness'].values()
])

numTrials = 10000
avgErr = np.zeros(5)
for i in range(numTrials): # can probably vectorize this loop
  if(i%1000 == 0):
    print i
  guess = np.random.rand(trueVal.shape[0], trueVal.shape[1])
  error = np.abs(guess - trueVal)
  avgErr = avgErr + np.mean(error, axis=1)

avgErr = avgErr / numTrials

print avgErr

# The last time I ran this script I got [ 0.27375645  0.27395781  0.27092006  0.2746411   0.27601202]

