"""
train.py
--------
Bu script, prepare_dataset.py'nin urettigi merged_dataset/ uzerinde
YOLO11 modelini egitir. Yani is bolumu soyle:

  prepare_dataset.py -> VERIYI hazirlar (indirme, sinif birlestirme, merged_dataset/)
  train.py            -> HAZIR VERIYI alip MODELI egitir

Bu ikisini ayri dosyalarda tutmamizin sebebi: veri hazirlama islemini
(uzun surebilir, indirme icerir) tekrar tekrar calistirmadan, sadece
egitim parametrelerini (epoch, batch, model boyutu vs.) degistirip
train.py'yi tek basina defalarca calistirabilmek.

NEDEN YOLO11 (YOLOv8 DEGIL)?
- Ultralytics'in ayni "ultralytics" kutuphanesi/API'si ile calisir;
  yani model.train(), model.predict() gibi tum kod YOLOv8 ile birebir
  aynidir, sadece agirlik dosyasi (.pt) degisir. Gecis maliyeti sifira
  yakin.
- YOLO11, YOLOv8'e gore genelde daha az parametreyle (daha hafif model)
  esdeger ya da daha iyi dogruluk (mAP) veriyor; yani hem daha hizli
  hem de en az o kadar isabetli.
- Ultralytics artik gelistirme/guncelleme agirligini YOLO11 (ve
  sonrasi) uzerine verdi; YOLOv8 hala calisir ama YOLO11 daha guncel.
NEDEN "s" (SMALL) VARYANTI (ilk denemedeki "n"/nano DEGIL)?
- Ilk egitim (50 epoch, yolo11n) analiz edildi: recall 0.52'de kaldi,
  yani model nesnelerin yarisini kaciriyor; ozellikle duman (smoke)
  gibi sekilsiz/yari saydam nesnelerde nano modelin ~2.6M parametresi
  kapasite sinirina takiliyor.
- yolo11s (~9.4M parametre) 6GB VRAM'li RTX 3060 Laptop GPU'ya
  batch=4 + AMP (otomatik karisik hassasiyet, varsayilan acik) ile
  sigar ve zor siniflarda belirgin mAP artisi saglar.
  (Not: batch=8 denendi, epoch ortasinda CUDA OOM verdi; laptop
  GPU'sunda 6GB'nin bir kismini ekran/Windows da kullaniyor.)
- Daha buyugu (m/l/x) bu GPU'da batch'i cok dusurur; kazanc/maliyet
  dengesi bozulur.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# merged_dataset/data.yaml: prepare_dataset.py tarafindan uretilen,
# ortak siniflari (0=fire, 1=smoke) tanimlayan dosya.
DATA_YAML = BASE_DIR / "merged_dataset" / "data.yaml"

# Baslangic agirligi: YOLO11'in "small" onceden egitilmis modeli.
# Kendi verimizle fine-tune edecegiz (transfer learning), sifirdan degil.
# Dosya klasorde yoksa Ultralytics ilk calistirmada otomatik indirir.
MODEL_WEIGHTS = BASE_DIR / "yolo11s.pt"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} bulunamadi. Once 'python prepare_dataset.py' "
            f"calistirip merged_dataset/ klasorunu olusturmalisin."
        )

    # GPU varsa onu kullan (cok daha hizli), yoksa CPU'ya dus.
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Egitim cihazi: {device.upper()}")

    model = YOLO(str(MODEL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=150,             # ilk denemede (50 epoch) loss/mAP egrileri hala yukseliyordu,
                                # yani egitim erken kesilmisti; ust siniri yukari cekiyoruz
        patience=30,            # erken durdurma: metrik 30 epoch boyunca iyilesmezse egitim
                                # kendiliginden durur, 150'yi beklemek zorunda kalmayiz
        imgsz=640,              # egitim goruntu boyutu; YOLO'nun standart varsayilani
        batch=4,                # yolo11s + 6GB VRAM'de batch=8 CUDA OOM verdi; 4 guvenli.
                                # Kucuk batch'in etkisini Ultralytics kendisi telafi eder:
                                # gradyanlari biriktirip efektif batch'i 64'e tamamlar (nbs=64)
        workers=4,              # 8 dataloader worker sistem RAM'ini zorluyordu; 4 yeterli
        optimizer="AdamW",      # SGD'ye gore genelde daha az elle ayar gerektirir, hizli yakinsar
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="fire_smoke_yolo11s",
    )

    print("[OK] Egitim tamamlandi.")


if __name__ == "__main__":
    main()
