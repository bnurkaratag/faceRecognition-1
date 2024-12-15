import cv2
import numpy as np
import os
import pickle

def train_model():
    print("Model eğitimi başlıyor...")
    
    # Cascade sınıflandırıcı
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Veri setinden yüzleri ve etiketleri al
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    print("Dataset klasörü taranıyor...")
    
    # Dataset klasöründeki her kişi için
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("Dataset klasörü bulunamadı!")
        return
        
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            print(f"İşleniyor: {person_name}")
            label_dict[current_label] = person_name
            
            # Kişinin her fotoğrafı için
            for img_name in os.listdir(person_path):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(person_path, img_name)
                    print(f"Fotoğraf okunuyor: {img_path}")
                    
                    # Görüntüyü oku ve gri tonlamaya çevir
                    image = cv2.imread(img_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Yüzü tespit et
                    face_rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    
                    for (x, y, w, h) in face_rects:
                        face_roi = gray[y:y+h, x:x+w]
                        # Yüzü standart boyuta getir
                        face_roi = cv2.resize(face_roi, (100, 100))
                        faces.append(face_roi)
                        labels.append(current_label)
                        
            current_label += 1
    
    print(f"Toplam {len(faces)} yüz tespit edildi.")
    
    if len(faces) == 0:
        print("Hiç yüz tespit edilemedi! Eğitim yapılamıyor.")
        return
        
    print("Model eğitiliyor...")
    
    # Yüz tanıyıcıyı oluştur ve eğit
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Modeli kaydet
    recognizer.save('trainer.yml')
    print("Model trainer.yml olarak kaydedildi.")
    
    # Etiket sözlüğünü kaydet
    with open('labels.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    print("Etiketler labels.pkl olarak kaydedildi.")
    
    print("\nModel eğitimi tamamlandı!")
    print(f"Tanınan kişiler: {list(label_dict.values())}")

if _name_ == "_main_":
    try:
        train_model()
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
