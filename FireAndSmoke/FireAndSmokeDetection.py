import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def find_latest_best_weights():
    # Sadece FireAndSmoke/runs/detect altindaki egitimlere bakar (baska
    # projelerdeki best.pt dosyalarina karismaz), en son degistirilen
    # (en guncel egitimden kalan) best.pt dosyasini otomatik secer.
    base_dir = Path(__file__).resolve().parent
    runs_dir = base_dir / "runs" / "detect"

    candidates = list(runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"{runs_dir} altinda hicbir 'best.pt' bulunamadi. Once train.py ile "
            f"bir egitim tamamlanmis olmali."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


class BatchImageProcessor:

    def __init__(self, model_path, conf_thresholds=None, use_tta=False):
        #sınıf başlatılıdığında modeli hafızaya yükler

        print(f"[BİLGİ] model yükleniyor, model: {model_path}\n")
        self.model = YOLO(model_path)

        # Sınıf BAZLI güven eşikleri: tek ortak eşik yerine her sınıfa ayrı eşik.
        # Neden? Duman (smoke) şekilsiz/yarı saydam olduğu için model ona doğal
        # olarak DÜŞÜK skor verir; ateş (fire) ise parlak/turuncu her şeyle
        # karışabildiği için yanlış alarma daha yatkındır. Bu yüzden:
        #   fire  -> yüksek eşik (yanlış alarmı azalt)
        #   smoke -> düşük eşik  (kaçırmayı azalt)
        if conf_thresholds is None:
            conf_thresholds = {"fire": 0.45, "smoke": 0.30}
        self.conf_thresholds = conf_thresholds

        # TTA (Test-Time Augmentation): görüntüyü birkaç farklı halde (ölçek,
        # ayna) modele sokup sonuçları birleştirir; ~2-3 kat yavaş ama recall
        # (bulma oranı) artar. Toplu fotoğraf işlerken açılabilir.
        self.use_tta = use_tta

        print(f"[BİLGİ] Sınıf eşikleri: {conf_thresholds}, TTA: {use_tta} ile model yüklendi.\n")
    
    def process_folder(self,input_folder_path,output_folder_path=None):
        #dosya dizinindeki fotoğrafların tamamını işler

        if not input_folder_path:
            print(f"[HATA] girdi olarak verilmesi gereken klasor yok\n")
            return
        
        if output_folder_path and not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=False)
            #yeni klasör oluşturur eğer varsa exist_ok=false olduğu için hata verir
    
        file_list = os.listdir(input_folder_path)
        #dosya dizinindeki dosyaları listeler

        valid_extensions = (".jpg", ".png", ".jpeg",".webp", ".tiff",".bmp")
        image_files = [f for f in file_list if f.lower().endswith(valid_extensions)]
        #dosya uzantılarını kontrol eder ve sadece resimleri alır
        
        print(f"[BİLGİ] Toplam {len(image_files)} adet fotoğraf işlenecek \n")

        for index, image_name in enumerate(image_files, start=1):
            img_path = os.path.join(input_folder_path,image_name)
            print(f"[{index}/{len(image_files)}] İşleniyor: {image_name}")

            # conf olarak eşiklerin EN DÜŞÜĞÜNÜ veriyoruz ki hiçbir sınıfın
            # adayı baştan elenmesin; asıl sınıf bazlı eleme aşağıda yapılıyor.
            # imgsz=640: modelin eğitildiği çözünürlükle aynı -> tutarlı skorlar.
            results = self.model.predict(
                source=img_path,
                conf=min(self.conf_thresholds.values()),
                imgsz=640,
                augment=self.use_tta,
                verbose=False,
            )
            #görüntüdeki nesneleri tahmin eder
            result = results[0]

            # Sınıf bazlı eşik filtresi: her kutunun sınıf adına bakıp
            # o sınıfın kendi eşiğinin altındaki kutuları eler.
            keep = [
                i for i in range(len(result.boxes))
                if float(result.boxes.conf[i])
                >= self.conf_thresholds.get(result.names[int(result.boxes.cls[i])], 0.5)
            ]
            result = result[keep]

            detected_count = len(result.boxes)
            print(f"Bulunan Tehlike Sayısı: {detected_count}")
            
            annotated_images = result.plot()
            #YOLO sayesinde tespit edilenlerin kutular çizdirildikten sonra dönmesini sağlar

            cv2.imshow("Fire & Smoke Detector - (Sonraki foto icin bir tusa bas, cikmak icin ESC)", annotated_images)

            if output_folder_path:
                save_path = os.path.join(output_folder_path, f"islenmis_{image_name}")
                cv2.imwrite(save_path, annotated_images)
                print(f"[KAYDEDİLDİ]")

            # waitKey ve ESC kontrolu artik output_folder_path bloğunun disinda:
            # kaydetme kapali olsa bile pencere beklesin ve ESC ile cikilabilsin
            # ESC disinda herhangi bir tusa basilirsa dongu bir sonraki fotografa gecer
            key = cv2.waitKey(0)

            if key == 27:
                print("işlem kullanıcı tarafından iptal edildi (ESC)")
                break
        cv2.destroyAllWindows()
        print("\n Tüm işlemler tamam")


if __name__ == "__main__":
    # FireAndSmoke/runs/detect altindaki en son egitimin best.pt'sini otomatik bulur;
    # elle klasor adi (fire_smoke_yolo11-4 gibi) yazmaya gerek kalmaz.
    MODEL_PATH = find_latest_best_weights()
    
    # 2. İşlenecek fotoğrafların bulunduğu klasör
    # Eğer bu kod (py dosyası) ile FireAndSmokeExamples klasörü yan yanaysa sadece ismini yazman yeterli.
    # Değilse, klasörün tam yolunu (C:\Users\...) yazmalısın.
    INPUT_FOLDER = r"C:\Users\Yavuz Altay\Desktop\İyex\FireAndSmoke\FireAndSmokeExamples"
    
    OUTPUT_FOLDER = r"C:\Users\Yavuz Altay\Desktop\İyex\FireAndSmoke\IslemGormusFotograflar"

    # Sınıfı çağır ve başlat
    # fire: 0.45 -> parlak/turuncu şeylere yanlış alarm vermesin diye yüksek
    # smoke: 0.30 -> duman skorları doğal olarak düşük olduğu için düşük
    # use_tta=True yaparsan daha çok nesne bulur ama ~2-3 kat yavaşlar.
    processor = BatchImageProcessor(
        model_path=MODEL_PATH,
        conf_thresholds={"fire": 0.45, "smoke": 0.30},
        use_tta=False,
    )
    
    # Klasör işleme sürecini başlat
    processor.process_folder(input_folder_path=INPUT_FOLDER, output_folder_path=OUTPUT_FOLDER)