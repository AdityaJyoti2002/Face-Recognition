# from os import name
# from contourpy import LineType
import face_recognition
import cv2
# from matplotlib import scale
import numpy as np
import datetime 
import csv

video_capture = cv2.VideoCapture(0)

aditya_image= face_recognition.load_image_file("/Users/adityajyoti/Documents/Python/project/face-recongnition/face/adityaprofile.jpeg")
aditya_encoding= face_recognition.face_encodings(aditya_image)[0]

known_face_encodings = [ aditya_image]
known_face_names = ['Aditya Jytoi']

# list of expect students
students = known_face_names.copy()

face_locations=[]
face_encoding =[]

# get current time 
Now = datetime.datetime.now()  
current_date = Now.strftime("%y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx = 0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # face_distance = face_recognition.face_distance(face_encodings, face_encoding)
        matches =face_recognition.compare_faces (known_face_encodings, face_encoding)

        face_distance = face_recognition.face_distance (known_face_encodings,face_encoding)
        best_matche_index = np.argmin(face_distance)

        if(matches[best_matche_index]):
            name = known_face_names[best_matche_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText =(10,100)
            fontscale = 1.5
            fontcolor = (255, 0, 0)
            thickness =3
            lineType = 2
            cv2.putText(frame, name +"Present", bottomLeftCornerOfText, font, fontscale, fontcolor, thickness,lineType )
            
            if name in students:
                students.remove(name)
                current_time= Now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

        cv2.imshow("Attendace", frame)
        if cv2.waitKey(1) & 0xFF== ord("q"):
            break

vedio_capture.release()
cv2.destroyAllWindows()
f.close()


