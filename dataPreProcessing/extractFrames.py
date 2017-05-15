import cv2
import os
import sys
import imageio
'''
VideoCapture variables of interest
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
    CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
'''

'''
More videocapture documentation at
http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-grab
'''


'''
a function that captures numCaptureFPS Frames per second
from a video given by vidCap - a cv2.VideoCapture object
'''
def getfps_windows(vidname, numcapturefps, saveto='./'):
    vidcap = cv2.videocapture(vidname)
    filenmtemplate = os.path.join(saveto, 'frame%d.jpg')

    numframes = vidcap.get(cv2.cap_prop_frame_count)
    print 'numframes ' + str(numframes)
    videofps = vidcap.get(cv2.cap_prop_fps)
    print 'videofps ' + str(videofps)
    videolength = numframes /videofps
    print 'videolength ' + str(videolength)
    vidcap.set(cv2.cap_prop_pos_frames, 0)

    # todo: remove line below
    videolength = 1 # just for now for debugging
    for i in range(videolength*numcapturefps):
       curframe = i*videofps/numcapturefps
       vidcap.set(cv2.cap_prop_pos_frames, curframe)
       success,image = vidcap.read()
       print 'read a new frame: ', success
       cv2.imwrite(filenmtemplate % i, image)     # save frame as jpeg file



'''
a function that captures numCaptureFPS Frames per second
from a video given by vidCap - a cv2.VideoCapture object
created after it looks like VideoCapture doesn't work on ubuntu
'''
# math.floor def getfps_ubuntu(vidname, numcapturefps, saveto='./'):
def getfps_ubuntu(vidname, numcapturefps, saveto):
    vid = imageio.get_reader(vidname,  'ffmpeg')
    filenmtemplate = os.path.join(saveto, 'frame%d.jpg')
 
    meta_data = vid.get_meta_data()
    numframes = meta_data["nframes"]
    print 'numframes ' + str(numframes)
    videofps = meta_data["fps"]
    print 'videofps ' + str(videofps)
    videolength = meta_data["duration"]
    print 'videolength ' + str(videolength)

    for i in range(int(videolength*numcapturefps)):
       curframe = i*int(videofps/numcapturefps)
       image = vid.get_data(curframe)
       imageio.imwrite(filenmtemplate % i, image) # save frame as jpeg file


'''
Performes the specified function on all mp4 files in a directory specified by path
Right now there is only getFPS, an make function template later
'''
def doToAllMoviesInDir(path):
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            vidName = os.path.join(path, file)
            newFileName = os.path.join(path, file[:-4]) # excluding .mp4 endign
            newFileName = newFileName + '_5fps' 
            print 'newFileName' +  newFileName
            os.system('sudo mkdir ' + newFileName)
            print 'vidname ' + vidName
            numFPS = 5
            getfps_ubuntu(vidName, numFPS, newFileName)
            # getfps_ubuntu(vidName, numFPS, saveTo=newFileName)
            # getFPS_windows(vidcap, numFPS, saveTo=newFileName)

if __name__ == "__main__":
    sourceFile = sys.argv[1]
    print 'sourceFile ' + sourceFile
    doToAllMoviesInDir(sourceFile)
