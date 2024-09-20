import cv2 as cv
import pickle
import tkinter as tk
from tkinter import Label, Button, Toplevel
from PIL import Image, ImageTk
import face_recognition
import numpy as np
from registration import FaceEncodingDatabase


class FaceRecognitionApp:
    def __init__(self, root):
        self.cap = cv.VideoCapture(0)
        self.root = root
        self.root.title("Arayüz")
        self.root.geometry("1100x500")

        with open("people_encoded.p", "rb") as file:
            self.names, self.people_encoded = pickle.load(file)

        
        self.resim_label = Label(self.root, bg="gray")
        self.resim_label.grid(row=0, column=0, padx=10, pady=10)

        self.isim_label = Label(self.root, text="", font=("Arial", 35), fg="green")
        self.isim_label.place(x=900, y=80)

        self.giris_button = Button(self.root, text="Giriş", width=10, height=2, bg="red", font=("Arial", 25),
                                   command=self.giris)
        self.giris_button.place(x=800, y=250)

        self.kayıt_button = Button(self.root, text="Yeni Kayıt", width=15, height=5, bg="red",
                                   command=self.create_new_user)
        self.kayıt_button.place(x=840, y=380)

        self.show_frame()

    def Load_people_from_df(self, db_path, output_file):
        db = FaceEncodingDatabase(db_path=db_path, output_file=output_file)
        db.process_and_save()

    def create_new_user(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            encoded = face_recognition.face_encodings(frame)

            def get_name():
                name = entry.get()
                yeni_pencere.destroy()
                if encoded:
                    self.names.append(name)
                    self.people_encoded.append(encoded[0])

            # Open a new window to get the user's name
            yeni_pencere = Toplevel(self.root)
            yeni_pencere.title("İsim Giriş")
            yeni_pencere.geometry("300x100")

            entry = tk.Entry(yeni_pencere, width=30)
            entry.pack(pady=20)

            button1 = tk.Button(yeni_pencere, text="Kaydet", command=get_name)
            button1.pack(pady=10)

    def new_page(self):
        new_window = Toplevel(self.root)
        new_window.title("Ana Sayfa")
        new_window.geometry("900x400")

        label = Label(new_window, text="Başarıyla Giriş Yaptınız", font=("Arial", 45), fg="green")
        label.pack(pady=50)

    def giris(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_cur_frame = face_recognition.face_locations(frame)
            cur_encoded = face_recognition.face_encodings(frame, face_cur_frame)

            if len(face_cur_frame) >= 1:
                for encodeFace, faceLoc in zip(cur_encoded, face_cur_frame):
                    matches = face_recognition.compare_faces(self.people_encoded, encodeFace)
                    distance = face_recognition.face_distance(self.people_encoded, encodeFace)
                    y1, x2, y2, x1 = faceLoc
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    match_id = np.argmin(distance)

                    if all(not item for item in matches):
                        self.isim_label.configure(text="Kişi Bulunamadı!", font=("Arial", 30), fg="red")
                        self.isim_label.place(x=770, y=60)

                    elif matches[match_id]:
                        self.isim_label.configure(text=f"Hoşgeldiniz...\n{self.names[match_id]}", fg="green",
                                                  font=("Arial", 35))
                        self.isim_label.place(x=760, y=60)
                        self.new_page()

            else:
                self.isim_label.configure(text="Yüz Algılanmadı!", font=("Arial", 30), fg="red")
                self.isim_label.place(x=750, y=60)

            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.resim_label.imgtk = imgtk
            self.resim_label.configure(image=imgtk)

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.resim_label.imgtk = imgtk
            self.resim_label.configure(image=imgtk)

        self.resim_label.after(10, self.show_frame)

    def on_close(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
