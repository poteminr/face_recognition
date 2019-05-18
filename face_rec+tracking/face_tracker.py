import cv2
import dlib
import time
import face_recognition
import pickle
import time
import scipy.misc
from subprocess import call
from FaceDB import FileDB



def send_unknown(full_image, camera, top, bottom, left, right, all_picture = False):
    
    t = time.localtime()
    t = time.asctime(t)
    
    text = "ALARM!!! "+" time: " +str(t)+ " " +camera
    
    if all_picture: 
        image = full_image[top:bottom, left:right]
        scipy.misc.imsave('unknown.jpg', image)
    else:
        scipy.misc.imsave('unknown.jpg', full_image)
    
    call('telegram-send ' + f'"{text}"', shell=True)
    call('telegram-send --file unknown.jpg', shell = True)


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data_roma.pickle', 'rb') as f:
     data = pickle.load(f)

known_face_names, known_face_encodings = data.get_data()

def doRecognizePerson(faceNames, fid, name):
    faceNames[fid] = name


database = FileDB('database.json')



def detectAndTrackMultipleFaces():
    cam = 'test_out_04.avi'
    capture = cv2.VideoCapture(cam)
    process_this_frame = True


    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow("result-image", 400, 100)

    cv2.startWindowThread()


    frameCounter = 0
    currentFaceID = 0

    faceTrackers = {}
    faceNames = {}

    try:
        while True:
            rc,fullSizeBaseImage = capture.read()

            baseImage = cv2.resize(fullSizeBaseImage, (0,0), fx = 0.6, fy = 0.6)
            baseImage = baseImage[:, :, ::-1]

            pressedKey = cv2.waitKey(5)
            if pressedKey == ord('Q'):
                break



   
            resultImage = baseImage.copy()




            frameCounter += 1



            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )


                if trackingQuality < 7:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )
                faceNames.pop(fid, None)



                #best param = 6
            if (frameCounter % 6) == 0:




                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)


                
                
                # y = top
                # x = left
                # h = bottom - y
                # w = right - x 



                
                
                # face_locations = face_recognition.face_locations(baseImage)
                face_locations = faceCascade.detectMultiScale(gray, 1.3, 5)
                # top, right, bottom, left

                

                fl = []
                for (_x,_y,_w,_h) in face_locations:

                    if (_w**2 + _h**2)**0.5 < 100:

                        fl.append((_x,_y,_w,_h))
                
                face_locations = fl
                del fl

                face_locations = [(_y, _x+_w, _y+_h, _x) for (_x,_y,_w,_h) in face_locations]

                face_encodings = face_recognition.face_encodings(baseImage, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        
                        
                    face_names.append(name)


                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    y = top
                    x = left
                    h = bottom - y
                    w = right - x 




                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h



 
                    matchedFid = None

       
                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())
                        

                        

                        #Считаем центр
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

      
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and
                             ( t_y <= y_bar   <= (t_y + t_h)) and
                             ( x   <= t_x_bar <= (x   + w  )) and
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid

                                        # Если нет трека, делаем новый
                    if matchedFid is None:
                  

                        print("Creating new tracker " + str(currentFaceID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x-10,
                                                            y-20,
                                                            x+w+10,
                                                            y+h+20))

                        faceTrackers[currentFaceID] = tracker

                        
                        faceNames[currentFaceID] = name

                        #telegram-bot

                            # if name == 'Unknown':
                            # send_unknown(baseImage, cam, top, bottom, left, right)




                        alarm_bool = (name == 'Unknown')

                        status_type = 'Common'

                                                
                        act = {'status':status_type,'name':name, 'alarm':str(alarm_bool)}
                        database.append_action(cam, act)

                        # Счетчик idшников
                        currentFaceID += 1





            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())


                top = t_y
                bottom = t_y + t_h
                right = t_x + t_w
                left = t_x


                


                cv2.rectangle(resultImage, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(resultImage, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX


                try:

                    cv2.putText(resultImage, faceNames[fid], (left + 2, bottom - 1), font, 0.4, (255, 255, 255), 1)
                except KeyError:
                                cv2.putText(resultImage, faceNames[fid], (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)




            resultImage = resultImage[:, :, ::-1]

            # Рисуем
            cv2.imshow("result-image", resultImage)







    except KeyboardInterrupt as e:
        pass

    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    detectAndTrackMultipleFaces()