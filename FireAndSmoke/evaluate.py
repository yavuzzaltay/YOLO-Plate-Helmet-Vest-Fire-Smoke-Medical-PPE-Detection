"""
evaluate.py
-----------
Egitilmis modeli TEST setinde olcer ve mAP / precision / recall raporlar.

Neden gerekli? "Model duzeldi mi?" sorusunu ornek fotograflara goz kararı
bakarak degil, SAYIYLA cevaplamak icin. Yeni bir egitimden once ve sonra
bu scripti calistirip metrikleri karsilastirirsin:

    python evaluate.py            -> en son egitimin best.pt'sini test setinde olcer

Onemli: split="test" kullaniyoruz. Egitim sirasinda gorulen val seti degil,
modelin HIC gormedigi test seti uzerinde olcum yapmak en durust sonuctur.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

# En son egitimin best.pt'sini bulan fonksiyonu tekrar yazmak yerine
# FireAndSmokeDetection.py'den yeniden kullaniyoruz (tek dogru kaynak).
from FireAndSmokeDetection import find_latest_best_weights

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "merged_dataset" / "data.yaml"


def main() -> None:
    weights = find_latest_best_weights()
    print(f"[BILGI] Olculen model: {weights}\n")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)

    # model.val() zaten ayrintili bir tablo basar (sinif bazli AP dahil);
    # altta ozet metrikleri ayrica yazdiriyoruz.
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",   # val degil TEST seti: model bu goruntuleri hic gormedi
        imgsz=640,      # egitimle ayni cozunurluk
        device=device,
    )

    print("\n===== OZET (test seti) =====")
    print(f"mAP50      : {metrics.box.map50:.4f}   (IoU 0.50'de ortalama isabet)")
    print(f"mAP50-95   : {metrics.box.map:.4f}   (daha siki olcut, kutu hassasiyeti dahil)")
    print(f"Precision  : {metrics.box.mp:.4f}   (bulduklarinin ne kadari dogru)")
    print(f"Recall     : {metrics.box.mr:.4f}   (gercek nesnelerin ne kadarini buldu)")


if __name__ == "__main__":
    main()
