import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
newdir=r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE"
current_directory = os.getcwd()
print("Current Directory:", current_directory)
os.chdir(newdir)
path=r"service_account_key.json"
# path=r"FACE_RECOGNITION_DATABASE/service_account_key.json"
cred = credentials.Certificate(path)
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-recognition-5bfe3-default-rtdb.firebaseio.com/",
    'storageBucket':'face-recognition-5bfe3.appspot.com'
})
# folder_path = r"C:\Users\KIIT\PycharmProjects\vedant_projects\FACE_RECOGNITION_DATABASE\Images"
import os
print("Current working directory:", os.getcwd())
folder_path = "Images"
path_list = os.listdir(folder_path)
img_list = []
students_id = []
for path in path_list:
    T_img = cv2.imread(os.path.join(folder_path, path))
    img_list.append(T_img)
    students_id.append(os.path.splitext(path)[0])

    fileName = f'{folder_path}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def findencodings(imageslist):
    encode_list = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

##encoding the known images
print("encoding started....")
encode_list_known = findencodings(img_list)
encode_list_known_withId = [encode_list_known, students_id]
print("encoding complete")

##Making pickle file
print(students_id)
encode_path=r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE\encodefile.p"
file = open(encode_path,'wb')
pickle.dump(encode_list_known_withId,file)
file.close()
print("file saved")
