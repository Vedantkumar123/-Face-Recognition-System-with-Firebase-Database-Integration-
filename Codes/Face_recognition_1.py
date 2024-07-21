import face_recognition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import cvzone
import os
import pickle
import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
from firebase_admin import firestore
path=r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant projects and works\ML_Deep_learning_projects\Deep_Learning_Projects\FACE_RECOGNITION_DATABASE\service_account_key.json"
cred = credentials.Certificate(path)
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-recognition-5bfe3-default-rtdb.firebaseio.com/",
    'storageBucket':'face-recognition-5bfe3.appspot.com'
    # 'storageBucket':'we-chat-e1d5c.appspot.com'
})
bucket = storage.bucket()

##image list creation
folder_mode_path = r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE\Modes"
mode_path_list = os.listdir(folder_mode_path)
img_Mode_list = []
for path in mode_path_list:
    temp_img=cv2.imread(os.path.join(folder_mode_path,path))
    img_Mode_list.append(cv2.resize(temp_img,(414,633)))
print(len(img_Mode_list))

##Main video cam setup
# path="http://192.168.48.50:8080/video"
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
background_path = r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE\Background\Background_13.jpg"
imgbackground = cv2.imread(background_path)
imgbackground = cv2.resize(imgbackground, (1280, 720))

##loading the pickle file
print("loading started...")
encode_path=r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE\encodefile.p"
file = open(encode_path,'rb')
encode_list_known_withId=pickle.load(file)
file.close()
encode_list_known,students_id=encode_list_known_withId
print("loading finished")
print(students_id)
id=-1
modetype=2
counter=0
img_person=[]
frame_count=0
print(students_id)
while True:
    ret,frame=cam.read()
    img=frame
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    frame_count+=1
    if frame_count%2==0:
        face_cur_Frame=face_recognition.face_locations(imgs,model='hog')
        encode_cur_frame=face_recognition.face_encodings(imgs,face_cur_Frame)

        imgbackground[172:172+480, 93:93+640] = img
        # imgbackground[45:45+633, 840:840+414] = img_Mode_list[modetype]
        if face_cur_Frame:
            for encodeface,faceloc in zip(encode_cur_frame,face_cur_Frame):
                matches = face_recognition.compare_faces(encode_list_known,encodeface)
                face_dis = face_recognition.face_distance(encode_list_known,encodeface)
                match_index = np.argmin(face_dis)
                # print("matches: ", matches)
                # print("Face_dis: ", face_dis)
                # print(match_index)
                # print()
                if matches[match_index]:
                    print("known face detected")
                    print(students_id[match_index])
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    bbox = 65+x1, 162+y1, x2-x1, y2-y1
                    imgbackground = cvzone.cornerRect(imgbackground,bbox,rt=0)
                    id=students_id[match_index]
                    if counter == 0:
                        counter = 1
                        modetype= 3

                if counter != 0:
                    if counter == 1:
                        #get the data
                        student_info = db.reference(f'People/{id}').get()
                        print(student_info)
                        #get the image
                        blob = bucket.get_blob(f'Images/{id}.jpg')
                        print(blob)
                        array=np.frombuffer(blob.download_as_string(),np.uint8)
                        img_person=cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                        #update data of face_scan_times
                        datetimeobject = datetime.strptime(student_info['last_time_recon'],'%Y-%m-%d %H:%M:%S')
                        seconds_elapsed = (datetime.now() - datetimeobject).total_seconds()
                        print(seconds_elapsed)
                        if seconds_elapsed>30:
                            ref=db.reference(f'People/{id}')
                            student_info['times_scanned']+=1
                            ref.child('times_scanned').set(student_info['times_scanned'])
                            ref.child('last_time_recon').set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            modetype=1
                            counter=0
                            # imgbackground[45:45 + 633, 840:840 + 414] = img_Mode_list[modetype]
                    if modetype != 1:
                        if 10<counter<20:
                            modetype=0
                            # imgbackground[45:45 + 633, 840:840 + 414] = img_Mode_list[modetype]
                        if counter<=10:
                            (w,h),_=cv2.getTextSize(student_info['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                            offset=(414-w)//2
                            img_person=cv2.resize(img_person,(250,270))
                            imgbackground[45:45 + 633, 840:840 + 414] = img_Mode_list[modetype]
                            imgbackground[152:152+270,920:920+250]=img_person
                            cv2.putText(imgbackground,str(student_info['name']),(840+offset,500),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
                        counter+=1
                        if counter>=20:
                            modetype = 2
                            counter = 0
                            student_info=[]
                            img_person=[]
                            # imgbackground[45:45 + 633, 840:840 + 414] = img_Mode_list[modetype]
        else:
            modetype=2
            counter=0
        cv2.imshow("background",imgbackground)
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break
cam.release()
