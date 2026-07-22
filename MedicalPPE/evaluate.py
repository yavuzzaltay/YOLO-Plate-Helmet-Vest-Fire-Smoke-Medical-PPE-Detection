# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Egitilmis tibbi PPE modelini TEST setinde olcer ve iki sey uretir:

1) mAP / precision / recall ozeti (FireAndSmoke/evaluate.py ile ayni mantik:
   "model iyi mi?" sorusu goz karariyla degil SAYIYLA cevaplanir).

2) SINIF BAZLI ESIK ONERISI — FireAndSmoke'ta elle yapilan isi otomatikler:
   her sinifin F1-Confidence egrisinde (metrics.box.f1_curve) F1'in zirve
   yaptigi guven degerini bulur ve MedicalPPEVideo.py'deki
   DEFAULT_THRESHOLDS sozlugune YAPISTIRMAYA HAZIR bicimde basar.
   (fire 0.37 / smoke 0.23 degerleri de ayni yontemle bulunmustu.)

Kullanim:
    python evaluate.py                      -> dataset/data.yaml ile olcer
    python evaluate.py yol/veri/data.yaml   -> baska bir data.yaml ile olcer
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from MedicalPPEVideo import find_weights

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = BASE_DIR / "dataset" / "data.yaml"


def main() -> None:
    data_yaml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} bulunamadi. Veri setini MedicalPPE/dataset/ altina koy "
            f"veya yolu argumanla ver: python evaluate.py yol/data.yaml"
        )

    weights = find_weights()
    print(f"[BILGI] Olculen model: {weights}\n")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)

    metrics = model.val(
        data=str(data_yaml),
        split="test",   # val degil TEST seti: model bu goruntuleri hic gormedi
        imgsz=640,      # egitim/izleme ile ayni cozunurluk -> tutarli skorlar
        device=device,
    )

    print("\n===== OZET (test seti) =====")
    print(f"mAP50      : {metrics.box.map50:.4f}   (IoU 0.50'de ortalama isabet)")
    print(f"mAP50-95   : {metrics.box.map:.4f}   (daha siki olcut, kutu hassasiyeti dahil)")
    print(f"Precision  : {metrics.box.mp:.4f}   (bulduklarinin ne kadari dogru)")
    print(f"Recall     : {metrics.box.mr:.4f}   (gercek nesnelerin ne kadarini buldu)")

    # ───── Sinif bazli F1-zirvesi esik onerisi ─────
    # f1_curve: (sinif_sayisi, 1000) — her sinif icin 1000 guven noktasinda F1.
    # px: o 1000 guven noktasinin degerleri (0..1).
    try:
        f1_curve = metrics.box.f1_curve
        px = metrics.box.px
        names = model.names
    except AttributeError:
        print("\n[UYARI] Bu ultralytics surumunde f1_curve/px erisilemedi; "
              "esik onerisi atlandi.")
        return

    print("\n===== SINIF BAZLI ESIK ONERISI (F1 zirvesi) =====")
    print("Asagidaki sozlugu MedicalPPEVideo.py -> DEFAULT_THRESHOLDS'a yapistir:\n")
    print("DEFAULT_THRESHOLDS = {")
    for i, name in names.items():
        if i >= len(f1_curve):
            continue
        best_j = int(f1_curve[i].argmax())
        best_conf = float(px[best_j])
        best_f1 = float(f1_curve[i][best_j])
        print(f'    "{name}": {best_conf:.2f},   # F1 zirvesi: {best_f1:.3f}')
    print("}")


if __name__ == "__main__":
    main()
