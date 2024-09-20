import os
import cv2 as cv
import face_recognition
import pickle


class FaceEncodingDatabase:
    def __init__(self, db_path, output_file):
        self.db_path = db_path
        self.output_file = output_file
        self.names = []
        self.people_encoded = []

    try:
        def encode_faces(self):
            for img_name in os.listdir(self.db_path):
                img_path = os.path.join(self.db_path, img_name)
                self.names.append(os.path.splitext(img_name)[0])
                print(f"Processing {img_path}...")
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(img)

                if face_encodings:
                    self.people_encoded.append(face_encodings[0])

        def save_to_file(self):
            data = [self.names, self.people_encoded]
            with open(self.output_file, "wb") as file:
                pickle.dump(data, file)

        def add_img(self, img_path):
            self.names.append(os.path.splitext(os.path.basename(img_path))[0])
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                self.people_encoded.append(face_encodings[0])
                self.save_to_file()
                print(f"Data saved to {self.output_file}")
                print(f"Total faces processed: {len(self.names)}")
                print(f"New face added: {self.names[-1]}")

    except Exception as e:
        print("Error during loading images")
        print(f"An error occurred: {e}")

    def process_and_save(self):
        self.encode_faces()
        self.save_to_file()
        print(f"Data saved to {self.output_file}")
        print(f"Total faces processed: {len(self.names)}")


if __name__ == "__main__":
    db = FaceEncodingDatabase(db_path="DB", output_file="people_encoded.p")
    db.process_and_save()
