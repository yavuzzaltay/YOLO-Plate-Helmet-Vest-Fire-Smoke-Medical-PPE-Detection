# -*- coding: utf-8 -*-
"""
train.py
--------
MedicalPPE/dataset/ altindaki veri setiyle YOLO11 modelini egitir.

NOT: Ilk model su an baska bir ortamda egitiliyor; bu script sonraki
egitimler (daha buyuk model, daha yuksek cozunurluk, veri seti
genisletme) icin hazir bir sablon. FireAndSmoke/train.py ile ayni
gerekcelerle ayni iskeleti kullanir.

VERI SETI NEREYE KONACAK?
  MedicalPPE/dataset/data.yaml   (train/val/test yollari + sinif adlari)
  Roboflow'dan "YOLOv11" (veya YOLOv8 — ayni format) olarak indirilen
  zip'i MedicalPPE/dataset/ icine acman yeterli.

MODEL SECIMI (bkz. README.md "Arastirma notlari"):
- yolo11s: FireAndSmoke deneyimiyle ayni gerekce — nano modeller
  (Sensors 2025 kiyaslamasinda 31k goruntude ancak %85.7 mAP50)
  eldiven/gozluk gibi KUCUK ve yari saydam nesnelerde kapasite
  sinirina takiliyor; small varyanti 6GB VRAM'e batch=4 ile sigiyor.
- Kok dizindeki yolo26n.pt ile de denenebilir (daha yeni mimari);
  MODEL_WEIGHTS yolunu degistirmen yeterli, API ayni.
- P2 (stride-4) tespit basligi DENENMEDI: kucuk nesneler icin klasik
  bir cozum ama Ultralytics bu surumde YOLO11 icin hazir bir
  "yolo11-p2.yaml" saglamiyor (yalnizca cls/obb/pose/seg varyantlari
  var) — elle mimari yaml'i yazmak MEAG-YOLO'da reddettigimiz ayni
  bakim yukunu getirir (bkz. README "Arastirma notlari"). Once
  imgsz/veri gibi ucuz yollari tuket, hala tikaniliyorsa gundeme al.

KUCUK NESNE + TEK KAYNAK SORUNU (2026-07-17 tanisi, gercek sayilarla):
En zayif 3 sinif — no-surgical-gloves (mAP50-95 0.195), no-surgical-cap
(0.253), no-facial-gear (0.304) — iki ayri sorunun kesisimi:
  1) KUCUK KUTU: bu 3 sinifin medyan kutu alani goruntunun sadece
     %0.9-1.6'si (ornek: no-facial-gear medyan %1.60 iken pozitif
     karsiligi face-shield %16.36 — 10 KAT daha kucuk). YOLO'nun
     640px'teki ozellik haritalari bu boyuttaki nesneleri ayirt etmekte
     zorlaniyor — klasik "kucuk nesne" sorunu.
  2) TEK KAYNAK: no-surgical-cap/no-surgical-gloves/no-facial-gear/
     no-medical-attire orneklerinin TAMAMI (1000+ ornek her biri) TEK
     bir Roboflow projesinden (ds2: ecrioobjectdetection1/meppe)
     geliyor; ds1/ds3 bu siniflara hic katki yapmiyor. Gorsel olarak
     incelendi: bu kaynak agirlikli olarak STUDYO/STOK FOTOGRAF tarzi
     (tek kisi, duz arka plan, kontrollu isik) — gercek hastane
     ortaminin (kamera acisi, kalabalik, hareket bulanikligi) cok az
     temsilcisi var. Test seti de AYNI kaynaktan geldigi icin mevcut
     mAP sayilari bile GERCEK dagilim kaymasini tam yansitmiyor olabilir.
  Etiket KALITESI incelendi (8'er ornek/sinif, kutu cizilerek
  goruldu) — makul ve tutarli, yani sorun etiketleme HATASI degil.

  ONCELIKLI COZUM SIRASI:
    a) imgsz 640->768 (bu dosyada asagida uygulandi) — ucretsiz, veri
       gerektirmiyor, dogrudan kucuk-kutu sorununu hedefliyor.
    b) Bu 3 sinif icin BASKA Roboflow projelerinden (Sensors 2025'in
       kullandigi GP/Final Work/Chef1 tarzi gercek mutfak/hastane
       ortami kaynaklari gibi) ek "no_*" ornegi bul — tek-kaynak
       bagimliligini kirar.
    c) Gercek hedef ortamdan (senin hastane/klinik ortamin) birkac
       fotograf/video ile GOZLE dogrula — test seti sayilari staged-
       photo onyargisi tasiyabilir, gercek sahnede farkli davranabilir.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

# Baslangic agirligi: YOLO11 small. Dosya yoksa Ultralytics otomatik indirir.
MODEL_WEIGHTS = BASE_DIR / "yolo11s.pt"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} bulunamadi. Veri setini (data.yaml + train/valid/test) "
            f"MedicalPPE/dataset/ klasorune yerlestir."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Egitim cihazi: {device.upper()}")

    model = YOLO(str(MODEL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=150,             # FireAndSmoke deneyimi: 50 epoch'ta egriler hala yukseliyordu
        patience=30,            # 30 epoch iyilesme yoksa erken durdur
        imgsz=768,              # 640'tan yukseltildi: no-surgical-cap/-gloves/no-facial-gear
                                # gibi kucuk-kutulu siniflarin (medyan alan %0.9-1.6) tespitini
                                # hedefliyor (bkz. dosya basi "KUCUK NESNE + TEK KAYNAK SORUNU")
        batch=12,               # TAHMIN DEGIL, OLCUM (2026-07-17, bu GPU'da fraction=0.05
                                # mini-kosularla): 768px'te batch=4 -> 2.72GB, batch=8 -> 3.25GB,
                                # batch=12 -> 4.75GB (rahat), batch=16 -> 6.06GB (sinirda, egitim
                                # ortasi OOM riski). Eski "6GB'da batch=4 bile riskli" bilgisi
                                # guncel torch/ultralytics ile GECERSIZ cikti. Buyuk fiziksel
                                # batch BatchNorm istatistiklerini de iyilestirir — kucultmek
                                # metrik dusururdu, tam tersini yapiyoruz.
        workers=4,
        optimizer="AdamW",
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="medical_ppe_yolo11s_768",
    )

    print("[OK] Egitim tamamlandi. Simdi evaluate.py ile esikleri kalibre et.")


if __name__ == "__main__":
    main()
