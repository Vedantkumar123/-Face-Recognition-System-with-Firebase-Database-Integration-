import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
path=r"C:\Users\KIIT\OneDrive\Desktop\Vedant_Official\vedant_projects\Deep learning\FACE_RECOGNITION_DATABASE\service_account_key.json"
cred = credentials.Certificate(path)
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-recognition-5bfe3-default-rtdb.firebaseio.com/"
})
ref = db.reference("People")
data = {
    "sample_image_1":
    {
        "name":"charles xavier",
        "major":"Computer science",
        "DOB":"2001:12:23",
        "last_time_recon":"2023-11-18 19:12:23",
        "times_scanned":0
    },
    "sample_image_2":
    {
        "name":"emily blunt",
        "major":"economics",
        "DOB":"2004:12:17",
        "last_time_recon":"2023-11-18 19:02:23",
        "times_scanned":0
    },
    "sample_image_3":
    {
        "name":"paul criminol",
        "major":"ecm",
        "DOB":"2002:12:23",
        "last_time_recon":"2023-11-18 19:09:23",
        "times_scanned":0
    },
    "sample_image_4":
    {
        "name":"Vedant Kumar",
        "major":"Computer science",
        "DOB":"2003:12:19",
        "last_time_recon":"2023-11-18 19:12:20",
        "times_scanned":0
    },
    "sample_image_5":
    {
        "name":"Manis saha",
        "major":"Computer science",
        "DOB":"2002:12:14",
        "last_time_recon":"2023-11-18 19:07:20",
        "times_scanned":0
    },
    "sample_image_6":
    {
        "name":"Shlok Sharma",
        "major":"Computer science",
        "DOB":"2002:12:14",
        "last_time_recon":"2023-11-18 19:07:20",
        "times_scanned":0
    },
    "sample_image_7":
    {
        "name":"Parag Vastrad",
        "major":"Computer science",
        "DOB":"2002:12:14",
        "last_time_recon":"2023-11-18 19:07:20",
        "times_scanned":0
    }

}

for key,value in data.items():
    ref.child(key).set(value)
