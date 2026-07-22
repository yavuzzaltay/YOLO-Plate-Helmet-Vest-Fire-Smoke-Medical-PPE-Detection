# -*- coding: utf-8 -*-
"""
İyex Tespit Demo — Streamlit arayüzü
=====================================
Bu dosya PlateDetection/ (plaka okuma) ve VestAndPlateDetection/ (baret &
yelek tespiti) klasörlerindeki hazır modelleri/pipeline'ları web üzerinden
denemek için bir arayüz sağlar. Asıl tespit/OCR mantığına DOKUNULMAZ; bu
dosya sadece PlateDetection/colab_local.py ve PlateDetection/video_plaka.py
içindeki fonksiyonları çağırıp sonuçları tarayıcıda gösterir.

Çalıştırma (PlateDetection klasöründeki venv, tüm bağımlılıklar orada kurulu):
    PlateDetection/.venv/Scripts/python.exe -m streamlit run app.py
"""

import csv
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
# Türkçe karakterli dizinlerde Windows/PyTorch/Streamlit uyumluluğu için göreceli yollar kullanıyoruz:
try:
    PLATE_DIR = Path(os.path.relpath(BASE_DIR / "PlateDetection", os.getcwd()))
    VEST_DIR = Path(os.path.relpath(BASE_DIR / "VestAndPlateDetection", os.getcwd()))
    FIRE_DIR = Path(os.path.relpath(BASE_DIR / "FireAndSmoke", os.getcwd()))
    MED_DIR = Path(os.path.relpath(BASE_DIR / "MedicalPPE", os.getcwd()))
except Exception:
    PLATE_DIR = Path("PlateDetection")
    VEST_DIR = Path("VestAndPlateDetection")
    FIRE_DIR = Path("FireAndSmoke")
    MED_DIR = Path("MedicalPPE")

sys.path.insert(0, str(PLATE_DIR))
sys.path.insert(0, str(VEST_DIR))
sys.path.insert(0, str(FIRE_DIR))
sys.path.insert(0, str(MED_DIR))

st.set_page_config(page_title="İyex Tespit Demo", layout="wide")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",".webp",".avif"}
VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


# ─────────────────────────── ORTAK YARDIMCI FONKSİYONLAR ───────────────────────────

def validate_image_upload(uploaded_file):
    """Yüklenen dosyanın gerçekten açılabilir bir görüntü olup olmadığını doğrular."""
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in IMAGE_EXTS:
        return False, f"Desteklenmeyen uzantı: {ext}. İzin verilenler: {', '.join(sorted(IMAGE_EXTS))}"
    data = uploaded_file.getvalue()
    try:
        img = Image.open(__import__("io").BytesIO(data))
        img.verify()
    except Exception as e:
        return False, f"Görüntü açılamadı / bozuk dosya: {e}"
    return True, ""


def validate_video_upload(uploaded_file):
    """Yüklenen dosyanın OpenCV ile açılıp en az bir kare okunabildiğini doğrular."""
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in VIDEO_EXTS:
        return False, f"Desteklenmeyen uzantı: {ext}. İzin verilenler: {', '.join(sorted(VIDEO_EXTS))}", None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    ok = cap.isOpened()
    frame_ok = False
    if ok:
        frame_ok, _ = cap.read()
    cap.release()
    if not ok or not frame_ok:
        os.unlink(tmp.name)
        return False, "Video açılamadı veya hiç kare okunamadı (bozuk/desteklenmeyen codec).", None
    return True, "", tmp.name


def save_upload_to_temp(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def ip_camera_input(key):
    """
    IP Webcam (telefon) adres girişi + kısa kurulum rehberi.
    Kullanıcı ana adresi yapıştırırsa (http://192.168.1.35:8080) akış
    yolu '/video' otomatik eklenir — IP Webcam MJPEG akışını bu yoldan verir.
    """
    with st.expander("📱 Telefon kurulumu (ilk sefer için)"):
        st.markdown(
            "1. Telefon ve bilgisayar **aynı Wi-Fi ağında** olmalı.\n"
            "2. Telefonda **IP Webcam** uygulamasını aç, listenin en altındaki "
            "**Start server**'a dokun.\n"
            "3. Ekranın altında `http://192.168.1.XX:8080` gibi bir adres çıkar — "
            "onu aşağıya yaz.\n"
            "4. İpucu: uygulamada *Video preferences → Video resolution* içinden "
            "**1280x720** seç; 4K akış işlemeyi çok yavaşlatır.\n"
            "5. Bağlantıyı denemek için aynı adresi PC tarayıcısında da açabilirsin."
        )
    # ------------------ SEÇENEK A: HTTP (Şu An Aktif) ------------------
    url = st.text_input("IP Webcam adresi", "http://192.168.1.100:8080", key=key).strip()
    if not url:
        return None
    if not url.startswith("http"):
        url = "http://" + url
    from urllib.parse import urlparse
    if urlparse(url).path in ("", "/"):
        url = url.rstrip("/") + "/video"
    return url

    # ------------------ SEÇENEK B: RTSP (Yorum Satırı - İleride Geçmek İçin) ------------------
    # # RTSP kameraya geçmek için yukarıdaki "SEÇENEK A" kodlarını yorum satırı yapıp 
    # # aşağıdaki kodların başındaki '#' işaretlerini kaldırabilirsiniz:
    # url = st.text_input("RTSP Kamera adresi", "rtsp://admin:admin123@192.168.1.100:554/stream1", key=key).strip()
    # if not url:
    #     return None
    # if not url.startswith("rtsp://"):
    #     url = "rtsp://" + url
    # return url


def check_ip_camera(url):
    """Kameraya bağlanıp gerçekten tek kare okunabiliyor mu diye bakar."""
    cap = cv2.VideoCapture(url)
    ok = cap.isOpened()
    frame_ok = False
    if ok:
        frame_ok, _ = cap.read()
    cap.release()
    return ok and frame_ok


def get_video_duration(video_path):
    """Videonun toplam süresini saniye cinsinden döner (okunamazsa None)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return frame_count / fps


def video_duration_slider(video_path, key):
    """Süre slider'ını videonun gerçek uzunluğuna göre sınırlar."""
    duration = get_video_duration(video_path) if video_path else None
    if duration:
        max_allowed = max(1, int(duration + 0.999))
        default_val = min(20, max_allowed)
        return st.slider(
            "İşlenecek video süresi (sn)", 1, max_allowed, default_val,
            key=f"{key}_{max_allowed}",
        )
    return st.slider("İşlenecek video süresi (sn)", 5, 120, 20, key=f"{key}_fallback")


# ─────────────────────────── HER MODELİN KENDİ DOĞRULUK METRİKLERİ ───────────────────────────
# Statik/sabit değer YOK: her sekmedeki "Metrikleri Hesapla" butonu basıldığı
# anda model.val() ile İLGİLİ TEST SETİNE karşı taze ölçüm yapılır. Böylece
# yeni bir eğitimden sonra kod değişmeden güncel sonuç görülür. Sonuç
# st.session_state'te tutulur ki sekmeler arası geçişte / diğer widget'lar
# tetiklediği rerun'larda kaybolmasın (yalnızca butona tekrar basılınca
# yeniden hesaplanır).

def _metrics_from_val(metrics, source: str) -> dict:
    """Genel özete ek olarak SINIF BAZLI değerleri de çıkarır.

    metrics.box (Ultralytics'in DetMetrics/Metric nesnesi) zaten sınıf
    başına precision/recall/AP dizilerini tutuyor (p, r, ap50, ap);
    biz sadece okuyup F1'i (2PR/(P+R)) her sınıf için türetiyoruz —
    aynı formülü genel özette de kullandığımız için tutarlı kalıyor.
    """
    box = metrics.box
    per_class = []
    for i, cls_idx in enumerate(box.ap_class_index):
        p, r = float(box.p[i]), float(box.r[i])
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class.append({
            "Sınıf": metrics.names[int(cls_idx)],
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        })
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "source": source,
        "per_class": per_class,
    }


def compute_plate_metrics() -> dict:
    """Plaka modelini GERÇEK üretim pipeline'ıyla ölçer — düz model.val() DEĞİL.

    NEDEN? Gerçek görsel modu (colab_local.process_single_image) tek geçişli
    düz bir YOLO çağrısı yapmıyor: detect_plate_boxes() üç farklı ölçek/eşikte
    (imgsz=1280/conf=0.25, 1920/0.10, 640/0.03) tarıyor, kutuları IoU ile
    tekilleştiriyor, geometri filtresi (en/boy oranı, min genişlik/yükseklik)
    uyguluyor; YOLO hiç bulamazsa detect_plate_classical() (klasik CV) devreye
    giriyor. Düz model.val() bunların hiçbirini yapmadığı için ya gerçekte
    yakalanan küçük/uzak plakaları kaçırmış (recall'ı olduğundan düşük), ya da
    geometri filtresinin eleyeceği yanlış kutuları saymış (precision'ı farklı)
    gösterirdi. Bu yüzden AYNI tespit fonksiyonlarını çağırıp yalnızca
    Ultralytics'in AP/precision/recall matematiğini (ap_per_class, box_iou —
    model.val()'ın kendi içinde kullandığı fonksiyonlar) bu özel tahminlere
    uyguluyoruz.

    NOT — CLAHE bu ölçüme dahil DEĞİL ve olmamalı: CLAHE yalnızca tespit
    SONRASI, kırpılmış plakaya, OCR okuması için uygulanıyor (bkz.
    colab_local.py process_plate_candidate). Kutu bulma başarısını (mAP/
    precision/recall) etkilemiyor; OCR metin doğruluğu ayrı bir ölçüttür ve
    bu fonksiyonun kapsamı dışında.
    """
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class, box_iou

    # DİKKAT: mutlak (.resolve()) yol şart — birazdan os.chdir(PLATE_DIR)
    # yapılınca PLATE_DIR göreceli bir string olduğu için (Türkçe karakterli
    # dizin uyumluluğu, bkz. dosya başı) göreceli kalsaydı "PlateDetection/
    # PlateDetection/..." diye YANLIŞLIKLA iç içe geçip glob'u sessizce
    # boş döndürürdü (bu hata gerçek testte yakalandı).
    test_dir = (PLATE_DIR / "Plate-Detection-2" / "test").resolve()
    img_dir, lbl_dir = test_dir / "images", test_dir / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} bulunamadı.")

    cwd = os.getcwd()
    os.chdir(PLATE_DIR)  # colab_local'ın best.pt/relatif yol varsayımlarıyla tutarlı olsun
    try:
        import colab_local
        model = YOLO("best.pt")
        device = "0" if torch.cuda.is_available() else "cpu"

        iouv = np.linspace(0.5, 0.95, 10)  # Ultralytics'in standart 10 IoU eşiği (mAP50-95 için)
        all_tp, all_conf = [], []
        n_gt_total = 0

        img_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        for img_path in img_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            # ─── Gerçek doğru kutular (YOLO formatı → piksel xyxy) ───
            gt_boxes = []
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                    gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            n_gt_total += len(gt_boxes)

            # ─── Gerçek pipeline'ın kendi tespiti (process_single_image ADIM 1 ile birebir aynı) ───
            valid_boxes, _ = colab_local.detect_plate_boxes(model, frame, device)
            if not valid_boxes:
                classical = colab_local.detect_plate_classical(frame)
                if classical:
                    valid_boxes = [(x1, y1, x2, y2, 0.0, ar) for (x1, y1, x2, y2, _, ar) in classical]
            if not valid_boxes:
                continue  # bu görüntüde hiç tahmin yok; kaçırılan GT'ler zaten recall'a yansır

            pred_boxes = torch.tensor([list(b[:4]) for b in valid_boxes], dtype=torch.float32)
            confs = [b[4] for b in valid_boxes]

            if gt_boxes:
                iou = box_iou(torch.tensor(gt_boxes, dtype=torch.float32), pred_boxes).numpy()  # (n_gt, n_pred)
            else:
                iou = np.zeros((0, len(valid_boxes)))

            # Ultralytics'in match_predictions'ıyla AYNI algoritma: her IoU eşiğinde
            # en yüksek IoU'dan başlayarak hem gerçek kutu hem tahmin başına TEK eşleşme.
            tp = np.zeros((len(valid_boxes), len(iouv)), dtype=bool)
            for ti, thr in enumerate(iouv):
                if iou.shape[0] == 0:
                    continue
                matches = np.array(np.nonzero(iou >= thr)).T  # [gt_idx, pred_idx] çiftleri
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        order = iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        matches = matches[order]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    tp[matches[:, 1], ti] = True

            all_tp.append(tp)
            all_conf.extend(confs)

        if not all_tp:
            raise ValueError("Test setinde hiç tahmin üretilemedi (gerçek pipeline hiçbir görüntüde plaka bulamadı).")

        tp_arr = np.concatenate(all_tp, axis=0)
        conf_arr = np.array(all_conf)
        pred_cls = np.zeros(len(conf_arr), dtype=int)   # tek sınıf: plaka
        target_cls = np.zeros(n_gt_total, dtype=int)

        _, _, p, r, f1, ap, *_ = ap_per_class(tp_arr, conf_arr, pred_cls, target_cls)
    finally:
        os.chdir(cwd)

    n_img = len(img_paths)
    metrics_row = {
        "Sınıf": "plaka", "Precision": float(p[0]), "Recall": float(r[0]),
        "F1": float(f1[0]), "mAP50": float(ap[0, 0]), "mAP50-95": float(ap[0].mean()),
    }
    return {
        "precision": float(p[0]),
        "recall": float(r[0]),
        "map50": float(ap[0, 0]),
        "map50_95": float(ap[0].mean()),
        "source": f"GERÇEK pipeline (çok geçişli tespit + geometri filtresi + klasik CV yedek), "
                  f"test seti {n_img} görüntü — az önce ölçüldü",
        "per_class": [metrics_row],
    }


def compute_vest_metrics() -> dict:
    """VestAndPlateDetection için yerelde test seti yok; checkpoint'e gömülü
    (eğitim sırasındaki son doğrulama) değerleri okur — model.val() koşmaz."""
    import torch
    weights = VEST_DIR / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} bulunamadı.")
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    tm = ckpt.get("train_metrics") or {}
    if not tm:
        raise ValueError("best.pt içinde train_metrics bulunamadı.")
    return {
        "precision": float(tm["metrics/precision(B)"]),
        "recall": float(tm["metrics/recall(B)"]),
        "map50": float(tm["metrics/mAP50(B)"]),
        "map50_95": float(tm["metrics/mAP50-95(B)"]),
        "source": "Eğitim sırasındaki son doğrulama (checkpoint'e gömülü) — "
                  "yerelde ayrı bir test seti yok",
    }


def compute_fire_metrics() -> dict:
    """FireAndSmoke modelini kendi test setinde (merged_dataset/test) ölçer."""
    from ultralytics import YOLO
    from FireAndSmokeVideo import find_latest_best_weights
    data_yaml = FIRE_DIR / "merged_dataset" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} bulunamadı.")
    model = YOLO(find_latest_best_weights())
    metrics = model.val(data=str(data_yaml), split="test", imgsz=640, plots=False, verbose=False)
    n_img = len(list((FIRE_DIR / "merged_dataset" / "test" / "images").glob("*")))
    return _metrics_from_val(metrics, f"Bağımsız test seti (merged_dataset/test, {n_img} görüntü) — az önce ölçüldü")


def compute_medical_metrics() -> dict:
    """MedicalPPE'nin GERÇEK fotoğraf hattını kendi test setinde ölçer.

    model.val() değil: üretimde fotoğraf TTA + dilimli tespit katmanlarından
    geçiyor; kullanıcıya gösterilen metrik de o hattın metriği olmalı
    (plaka modülündeki ilkenin aynısı). Ölçüm matematiği yine Ultralytics'in
    (ap_per_class) — ayrıntı MedicalPPEVideo.evaluate_pipeline'da.
    """
    from ultralytics import YOLO
    from MedicalPPEVideo import find_weights, evaluate_pipeline
    img_dir = MED_DIR / "dataset" / "test" / "images"
    lbl_dir = MED_DIR / "dataset" / "test" / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"{img_dir} bulunamadı. Önce MedicalPPE/prepare_dataset.py çalıştırılmalı."
        )
    model = YOLO(find_weights())
    bar = st.progress(0.0, text="Gerçek hat (TTA + dilimli) test setinde ölçülüyor...")
    result = evaluate_pipeline(
        model, img_dir, lbl_dir,
        progress_cb=lambda done, total: bar.progress(
            done / total, text=f"Gerçek hat ölçülüyor... {done}/{total} görüntü"),
    )
    bar.empty()
    n_img = len(list(img_dir.glob("*")))
    result["source"] = (
        f"Bağımsız test seti ({n_img} görüntü) — GERÇEK hat (TTA + dilimli tespit) "
        f"ile az önce ölçüldü"
    )
    return result


MODEL_METRIC_COMPUTERS = {
    "plate": compute_plate_metrics,
    "vest": compute_vest_metrics,
    "fire": compute_fire_metrics,
    "medical": compute_medical_metrics,
}


def render_model_metrics(model_key):
    """'Metrikleri Hesapla' butonu + (varsa) son ölçüm sonucunu 5 sütunda gösterir."""
    state_key = f"metrics_{model_key}"
    if st.button("📊 Metrikleri Hesapla / Test Et", key=f"{model_key}_metrics_btn"):
        with st.spinner("Test seti üzerinde ölçülüyor (biraz sürebilir)..."):
            try:
                st.session_state[state_key] = MODEL_METRIC_COMPUTERS[model_key]()
            except Exception as e:
                st.session_state[state_key] = None
                st.error(f"Metrik hesaplanamadı: {e}")

    m = st.session_state.get(state_key)
    if m:
        p, r = m["precision"], m["recall"]
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        cols = st.columns(5)
        cols[0].metric("Precision", f"{p * 100:.1f}%")
        cols[1].metric("Recall", f"{r * 100:.1f}%")
        cols[2].metric("F1", f"{f1 * 100:.1f}%")
        cols[3].metric("mAP50", f"{m['map50'] * 100:.1f}%")
        cols[4].metric("mAP50-95", f"{m['map50_95'] * 100:.1f}%")
        st.caption(m["source"])

        per_class = m.get("per_class")
        if per_class:
            with st.expander(f"Sınıf bazlı değerler ({len(per_class)} sınıf)"):
                df = pd.DataFrame(per_class).set_index("Sınıf")
                for col in ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]:
                    df[col] = (df[col] * 100).round(1)
                st.dataframe(
                    df.style.format("{:.1f}%"),
                    width='stretch',
                )
        else:
            # Vest gibi checkpoint'ten okunan modellerde sınıf bazlı veri yok
            # (train_metrics yalnızca genel özeti saklıyor).
            st.caption("Sınıf bazlı değerler bu model için mevcut değil.")


# ─────────────────────────── PLAKA TESPİTİ: MODEL YÜKLEME ───────────────────────────

@st.cache_resource(show_spinner="Plaka modeli ve OCR motoru yükleniyor (ilk seferde biraz sürer)...")
def load_plate_pipeline():
    cwd = os.getcwd()
    os.chdir(PLATE_DIR)  # colab_local.initialize_model() "best.pt" dosyasını cwd'de arıyor
    try:
        import colab_local
        model = colab_local.initialize_model()
        reader, gpu = colab_local.initialize_ocr()
    finally:
        os.chdir(cwd)
    return colab_local, model, reader, gpu


@st.cache_resource(show_spinner="Baret/yelek modeli yükleniyor...")
def load_vest_model():
    from ultralytics import YOLO
    return YOLO(str(VEST_DIR / "best.pt"))


@st.cache_resource(show_spinner="Yangın/duman modeli yükleniyor...")
def load_fire_model():
    """FireAndSmoke/runs/detect altındaki EN SON eğitimin best.pt'sini yükler.

    Modeli bir kez yükleyip cache'liyoruz; her izleme oturumu için yeni bir
    FireSmokeMonitor kurulur ama hepsi bu tek model nesnesini paylaşır
    (izleyici kurulurken tracker hafızası sıfırlanır, detay FireAndSmokeVideo.py'de).
    """
    from ultralytics import YOLO
    from FireAndSmokeVideo import find_latest_best_weights
    return YOLO(find_latest_best_weights())


def medical_weights_or_none():
    """Tıbbi PPE model dosyası varsa yolunu, yoksa None döner.

    Cache'lenmez ve her rerun'da çalışır (ucuz bir glob): model henüz
    eğitimde olduğu için dosya SONRADAN gelecek; kullanıcı best.pt'yi
    MedicalPPE/ klasörüne koyup sayfayla etkileşime geçtiği anda sekme
    kendiliğinden aktifleşsin istiyoruz.
    """
    from MedicalPPEVideo import find_weights
    try:
        return find_weights()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Tıbbi PPE modeli yükleniyor...")
def load_medical_model(weights_path):
    """Ağırlık YOLUNA göre cache'ler: yeni bir eğitim (yeni yol) gelirse
    eski cache'e takılmadan yeni model yüklenir."""
    from ultralytics import YOLO
    return YOLO(weights_path)


# ─────────────────────────── PLAKA: GÖRSEL İŞLEME ───────────────────────────

def run_plate_image(colab_local, model, reader, image_path):
    debug_dir = tempfile.mkdtemp(prefix="plate_debug_")
    results = colab_local.process_single_image(image_path, model, reader, output_dir=debug_dir, save_debug=True)
    return results, debug_dir


def show_plate_image_debug(debug_dir, stem):
    """process_single_image'in ürettiği ara adım görüntülerinden en anlamlı ikisini gösterir."""
    import glob as _glob
    crop = _glob.glob(os.path.join(debug_dir, f"{stem}*_01_original_crop.jpg"))
    enhanced = _glob.glob(os.path.join(debug_dir, f"{stem}*_06_enhanced.jpg"))
    cols = st.columns(2)
    if crop:
        cols[0].image(crop[0], caption="Tespit edilen plaka kırpımı", width='stretch')
    if enhanced:
        cols[1].image(enhanced[0], caption="OCR için iyileştirilmiş görüntü", width='stretch')


# ─────────────────────────── PLAKA: VİDEO GERÇEK ZAMANLI ───────────────────────────

def run_plate_video(video_path, model, reader, max_seconds, frame_ph, status_ph, table_ph, csv_path, evidence_dir, live=False):
    import torch
    import video_plaka as vp

    device = "0" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Video açılamadı." if not live else
                 "Kameraya bağlanılamadı — telefon ve PC aynı Wi-Fi ağında mı? Adres doğru mu?")
        return

    if live:
        # Canlı akışta OpenCV varsayılan olarak birkaç kareyi tamponda
        # bekletir; işleme yavaş kalınca tampondaki ESKİ kareler işlenir
        # ve görüntü gitgide geriden gelir. Tamponu 1 kareye indirerek
        # her zaman en güncel kareye yakın çalışırız.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgsz = vp.pick_imgsz(frame_w)
    os.makedirs(evidence_dir, exist_ok=True)

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Plaka", "Tarih_Saat", "Guven_%", "Video_sn"])

    active_tracks = []
    logged = {}
    recent_plates = []
    sample_idx = 0
    frame_no = -1
    t0 = time.time()

    # Canlı modda örnekleme kare numarasıyla değil ZAMANLA yapılır:
    # dosyada "her 6. kare" deterministiktir ama canlı akışta kareler
    # bize ağın verdiği hızda gelir — 0.25 sn'de bir örnek (=4 Hz,
    # dosya modundaki FRAME_SKIP oranıyla aynı mertebe) hedeflenir.
    SAMPLE_PERIOD = 0.25
    last_sample = -SAMPLE_PERIOD

    def refresh_table():
        try:
            table_ph.dataframe(pd.read_csv(csv_path), width='stretch')
        except pd.errors.EmptyDataError:
            pass

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if live:
                # grab() burada iki iş görür: (1) yeni kare çeker,
                # (2) örnek zamanı gelmemişse kareyi DECODE ETMEDEN
                # çöpe atar → tampon boşalır, hep güncel kalırız.
                video_sec = time.time() - t0
                if video_sec > max_seconds:
                    break
                if video_sec - last_sample < SAMPLE_PERIOD:
                    continue
                last_sample = video_sec
            else:
                frame_no += 1
                if frame_no / fps > max_seconds:
                    break
                if frame_no % vp.FRAME_SKIP != 0:
                    continue
                video_sec = frame_no / fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            sample_idx += 1

            detections = vp.detect_boxes_fast(model, frame, device, imgsz)
            matched_tracks = set()
            for det in detections:
                box = det[:4]
                best_track, best_iou = None, vp.IOU_MATCH
                for tr in active_tracks:
                    if tr.id in matched_tracks:
                        continue
                    iou = vp._iou(box, tr.box)
                    if iou > best_iou:
                        best_track, best_iou = tr, iou
                if best_track is None:
                    best_track = vp.PlateTrack(box, sample_idx, video_sec)
                    active_tracks.append(best_track)
                else:
                    best_track.update(box, sample_idx, video_sec)
                matched_tracks.add(best_track.id)
                best_track.add_crop(frame, box, det[4], video_sec)

                if not best_track.resolved:
                    n = len(best_track.crops)
                    first_try = best_track.last_attempt_count == 0 and n >= 3
                    if first_try or n - best_track.last_attempt_count >= 2:
                        best_track.last_attempt_count = n
                        result = vp.attempt_read(best_track, reader, evidence_dir, "Canlı kontrol")
                        if result and result["cross_validated"]:
                            best_track.resolved = True
                            best_track.resolved_text = result["text"]
                            written = vp.log_result(best_track, result, logged, csv_writer, csv_file, evidence_dir)
                            if written:
                                recent_plates.append(written)
                                refresh_table()

            still_active = []
            for tr in active_tracks:
                if sample_idx - tr.last_seen > vp.MISS_LIMIT:
                    if not tr.resolved:
                        result = vp.attempt_read(tr, reader, evidence_dir, "İz kapandı")
                        written = vp.log_result(tr, result, logged, csv_writer, csv_file, evidence_dir) if result else None
                        if written:
                            recent_plates.append(written)
                            refresh_table()
                else:
                    still_active.append(tr)
            active_tracks = still_active

            elapsed = max(0.1, time.time() - t0)
            proc_fps = sample_idx / elapsed
            view = vp._draw_overlay(frame, active_tracks, recent_plates, video_sec, video_sec / elapsed, proc_fps)
            frame_ph.image(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), channels="RGB")
            sure_metni = f"{video_sec:5.1f}s" if live else f"{video_sec:5.1f}s / {max_seconds:.0f}s"
            status_ph.text(
                f"Video: {sure_metni}   |   "
                f"aktif iz: {len(active_tracks)}   |   loglanan plaka: {len(logged)}"
            )

        # Video bitti (veya süre sınırına ulaşıldı): hâlâ açık kalan ve
        # MISS_LIMIT'e hiç ulaşmamış izler için son bir OCR denemesi
        # yapılır — yoksa video kısa kesildiğinde tam önümüzdeki bir
        # plaka hiç işlenmeden kaybolur.
        for tr in active_tracks:
            if not tr.resolved:
                result = vp.attempt_read(tr, reader, evidence_dir, "Video bitti")
                if result:
                    written = vp.log_result(tr, result, logged, csv_writer, csv_file, evidence_dir)
                    if written:
                        recent_plates.append(written)
                        refresh_table()
    finally:
        cap.release()
        csv_file.close()

    kaynak = "Canlı yayın" if live else "Video"
    st.success(f"{kaynak} işleme tamamlandı — {len(logged)} farklı plaka CSV'ye yazıldı.")


# ─────────────────────────── BARET & YELEK: GÖRSEL / VİDEO ───────────────────────────

VEST_VIOLATION_CLASSES = {"no-hardhat", "no-safety-vest"}


def run_vest_image(vest_model, image_bgr, conf):
    results = vest_model.predict(source=image_bgr, conf=conf, verbose=False)
    r = results[0]
    annotated = r.plot()
    names = vest_model.names
    counts = {}
    if r.boxes is not None:
        for c in r.boxes.cls.tolist():
            cname = names[int(c)]
            counts[cname] = counts.get(cname, 0) + 1
    return annotated, counts


def log_vest_result(csv_path, counts, video_sec=None):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Tarih_Saat", "Video_sn", "kisi", "baret_var", "baret_yok", "yelek_var", "yelek_yok"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{video_sec:.1f}" if video_sec is not None else "",
            counts.get("person", 0),
            counts.get("hardhat", 0),
            counts.get("no-hardhat", 0),
            counts.get("safety-vest", 0),
            counts.get("no-safety-vest", 0),
        ])


def _draw_corner_info(frame, proc_fps):
    """Sağ üst köşeye kaynak çözünürlüğü + gerçek işlem hızını (FPS) yazar."""
    H, W = frame.shape[:2]
    text = f"{W}x{H}  |  {proc_fps:.1f} FPS (islem)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (W - tw - 16, 0), (W, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, text, (W - tw - 8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def run_vest_video(video_path, vest_model, conf, max_seconds, frame_ph, status_ph, table_ph, csv_path, frame_skip=5, live=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Video açılamadı." if not live else
                 "Kameraya bağlanılamadı — telefon ve PC aynı Wi-Fi ağında mı? Adres doğru mu?")
        return

    if live:
        # Canlı akışta tamponu küçült — eski kare birikmesin (detaylı
        # açıklama run_plate_video içinde)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if os.path.exists(csv_path):
        os.remove(csv_path)
    frame_no = -1
    sample_idx = 0
    violation_frames = 0
    t0 = time.time()

    # Canlı modda zaman tabanlı örnekleme (run_plate_video ile aynı mantık)
    SAMPLE_PERIOD = 0.25
    last_sample = -SAMPLE_PERIOD

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if live:
                video_sec = time.time() - t0
                if video_sec > max_seconds:
                    break
                if video_sec - last_sample < SAMPLE_PERIOD:
                    continue
                last_sample = video_sec
            else:
                frame_no += 1
                if frame_no / fps > max_seconds:
                    break
                if frame_no % frame_skip != 0:
                    continue
                video_sec = frame_no / fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            sample_idx += 1

            annotated, counts = run_vest_image(vest_model, frame, conf)
            violations = sum(counts.get(c, 0) for c in VEST_VIOLATION_CLASSES)
            if violations > 0:
                violation_frames += 1
                log_vest_result(csv_path, counts, video_sec)
                try:
                    table_ph.dataframe(pd.read_csv(csv_path), width='stretch')
                except pd.errors.EmptyDataError:
                    pass

            proc_fps = sample_idx / max(0.1, time.time() - t0)
            annotated = _draw_corner_info(annotated, proc_fps)
            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            sure_metni = f"{video_sec:5.1f}s" if live else f"{video_sec:5.1f}s / {max_seconds:.0f}s"
            status_ph.text(f"Video: {sure_metni}   |   ihlal tespit edilen kare: {violation_frames}")
    finally:
        cap.release()

    kaynak = "Canlı yayın" if live else "Video"
    st.success(f"{kaynak} işleme tamamlandı — {violation_frames} karede ihlal tespit edildi.")


# ─────────────────────────── YANGIN & DUMAN: VİDEO ───────────────────────────

def run_fire_video(video_path, izleyici, max_seconds, frame_ph, status_ph, table_ph, csv_path, frame_skip=5, live=False):
    """Video/canlı akışta yangın-duman izleme döngüsü.

    Kare okuma/örnekleme iskeleti run_vest_video ile birebir aynı (grab/retrieve,
    canlıda zaman bazlı örnekleme, dosyada kare atlama). Fark: her örnek kare
    FireAndSmokeVideo.FireSmokeMonitor'a verilir; o da takip (track) +
    N-of-M zamansal onay + kanıt fotoğrafı/CSV işlerini kendi içinde halleder.
    Alarm/bildirim yoktur — onaylanan olaylar sadece kaydedilir ve tabloda görünür.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Video açılamadı." if not live else
                 "Kameraya bağlanılamadı — telefon ve PC aynı Wi-Fi ağında mı? Adres doğru mu?")
        return

    if live:
        # Canlı akışta tamponu küçült — eski kare birikmesin (detaylı
        # açıklama run_plate_video içinde)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_no = -1
    sample_idx = 0
    olay_sayisi = 0
    t0 = time.time()

    # Canlı modda zaman tabanlı örnekleme (run_plate_video ile aynı mantık)
    SAMPLE_PERIOD = 0.25
    last_sample = -SAMPLE_PERIOD

    def refresh_table():
        try:
            table_ph.dataframe(pd.read_csv(csv_path), width='stretch')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if live:
                video_sec = time.time() - t0
                if video_sec > max_seconds:
                    break
                if video_sec - last_sample < SAMPLE_PERIOD:
                    continue
                last_sample = video_sec
            else:
                frame_no += 1
                if frame_no / fps > max_seconds:
                    break
                if frame_no % frame_skip != 0:
                    continue
                video_sec = frame_no / fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            sample_idx += 1

            # Tüm izleme zekâsı (track + N-of-M + kanıt kaydı) tek çağrıda:
            cizili, yeni_olaylar = izleyici.process_frame(frame, video_sec)
            if yeni_olaylar:
                olay_sayisi += len(yeni_olaylar)
                refresh_table()

            proc_fps = sample_idx / max(0.1, time.time() - t0)
            cizili = _draw_corner_info(cizili, proc_fps)
            frame_ph.image(cv2.cvtColor(cizili, cv2.COLOR_BGR2RGB), channels="RGB")
            ozet = izleyici.get_status()
            sure_metni = f"{video_sec:5.1f}s" if live else f"{video_sec:5.1f}s / {max_seconds:.0f}s"
            status_ph.text(
                f"Video: {sure_metni}   |   aktif iz: {ozet['active_tracks']}   |   "
                f"onaylı olay: {olay_sayisi}"
            )
    finally:
        cap.release()

    kaynak = "Canlı yayın" if live else "Video"
    st.success(f"{kaynak} işleme tamamlandı — {olay_sayisi} onaylı yangın/duman olayı kaydedildi.")


# ─────────────────────────── TIBBİ PPE: VİDEO ───────────────────────────

def run_medical_video(video_path, izleyici, max_seconds, frame_ph, status_ph, table_ph, csv_path, frame_skip=5, live=False):
    """Video/canlı akışta tıbbi PPE uyum izleme döngüsü.

    İskelet run_fire_video ile birebir aynı (grab/retrieve, canlıda zaman
    bazlı örnekleme, dosyada kare atlama). Fark: kareler
    MedicalPPEVideo.MedicalPPEMonitor'a verilir; o da kişi takibi +
    ekipman eşleştirme + N-of-M zamansal onay + kanıt kaydını kendi
    içinde halleder. Alarm/bildirim yoktur — onaylanan ihlaller sadece
    kaydedilir ve tabloda görünür.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Video açılamadı." if not live else
                 "Kameraya bağlanılamadı — telefon ve PC aynı Wi-Fi ağında mı? Adres doğru mu?")
        return

    if live:
        # Canlı akışta tamponu küçült — eski kare birikmesin (detaylı
        # açıklama run_plate_video içinde)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_no = -1
    sample_idx = 0
    ihlal_sayisi = 0
    t0 = time.time()

    # Canlı modda zaman tabanlı örnekleme (run_plate_video ile aynı mantık)
    SAMPLE_PERIOD = 0.25
    last_sample = -SAMPLE_PERIOD

    def refresh_table():
        try:
            table_ph.dataframe(pd.read_csv(csv_path), width='stretch')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if live:
                video_sec = time.time() - t0
                if video_sec > max_seconds:
                    break
                if video_sec - last_sample < SAMPLE_PERIOD:
                    continue
                last_sample = video_sec
            else:
                frame_no += 1
                if frame_no / fps > max_seconds:
                    break
                if frame_no % frame_skip != 0:
                    continue
                video_sec = frame_no / fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            sample_idx += 1

            # Tüm izleme zekâsı (track + eşleştirme + N-of-M + kanıt) tek çağrıda:
            cizili, yeni_ihlaller = izleyici.process_frame(frame, video_sec)
            if yeni_ihlaller:
                ihlal_sayisi += len(yeni_ihlaller)
                refresh_table()

            proc_fps = sample_idx / max(0.1, time.time() - t0)
            cizili = _draw_corner_info(cizili, proc_fps)
            frame_ph.image(cv2.cvtColor(cizili, cv2.COLOR_BGR2RGB), channels="RGB")
            ozet = izleyici.get_status()
            sure_metni = f"{video_sec:5.1f}s" if live else f"{video_sec:5.1f}s / {max_seconds:.0f}s"
            status_ph.text(
                f"Video: {sure_metni}   |   aktif iz: {ozet['active_tracks']}   |   "
                f"onaylı ihlal: {ihlal_sayisi}"
            )
    finally:
        cap.release()

    kaynak = "Canlı yayın" if live else "Video"
    st.success(f"{kaynak} işleme tamamlandı — {ihlal_sayisi} onaylı PPE ihlali kaydedildi.")


# ═══════════════════════════════════════ ARAYÜZ ═══════════════════════════════════════

st.title("İyex Tespit Demo")
st.caption("Plaka okuma ve baret/yelek tespiti modellerini tarayıcıdan deneyin.")

tab_plate, tab_vest, tab_fire, tab_med = st.tabs(
    ["🚗 Plaka Tespiti", "🦺 Baret & Yelek Tespiti", "🔥 Yangın & Duman Tespiti", "🧑‍⚕️ Tıbbi PPE"]
)

# ───────────────────────────────── PLAKA TESPİTİ SEKMESİ ─────────────────────────────────
with tab_plate:
    render_model_metrics("plate")
    sub_img, sub_video = st.tabs(["Görsel", "Video (gerçek zamanlı)"])

    with sub_img:
        st.subheader("Görselden plaka oku")
        source = st.radio(
            "Kaynak",
            ["Örnek klasördeki görselleri kullan (arabalar/)", "Kendi görselini yükle"],
            key="plate_img_source",
        )

        if source == "Örnek klasördeki görselleri kullan (arabalar/)":
            arabalar_dir = PLATE_DIR / "arabalar"
            samples = sorted(p for p in arabalar_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not samples:
                st.warning(f"'{arabalar_dir}' klasöründe örnek görsel bulunamadı.")
            else:
                st.caption(f"'{arabalar_dir}' klasöründeki {len(samples)} görsel:")
                preview_cols = st.columns(6)
                for i, p in enumerate(samples):
                    preview_cols[i % 6].image(str(p), caption=p.name, width='stretch')

                if st.button(f"Tüm örnek görselleri işle ({len(samples)} adet)", key="plate_batch_run"):
                    colab_local, model, reader, gpu = load_plate_pipeline()
                    if model is None:
                        st.error("Model yüklenemedi (best.pt bulunamadı).")
                    else:
                        with st.spinner(f"{len(samples)} görsel işleniyor, biraz sürebilir..."):
                            all_results = colab_local.process_all_images(str(arabalar_dir), model, reader, save_debug=True)
                        st.session_state["last_plate_results"] = all_results
                        st.session_state["last_plate_debug_dir"] = str(arabalar_dir / "sonuclar")
        else:
            uploaded = st.file_uploader("Görsel yükle", type=sorted(e.strip(".") for e in IMAGE_EXTS), key="plate_img_upload")
            image_path = None
            if uploaded is not None:
                ok, msg = validate_image_upload(uploaded)
                if not ok:
                    st.error(msg)
                else:
                    image_path = save_upload_to_temp(uploaded)
                    st.image(image_path, caption="Yüklenen görsel", width=400)

            if image_path and st.button("Plakayı Oku", key="plate_img_run"):
                colab_local, model, reader, gpu = load_plate_pipeline()
                if model is None:
                    st.error("Model yüklenemedi (best.pt bulunamadı).")
                else:
                    with st.spinner("İşleniyor (tespit → temizleme → OCR)..."):
                        results, debug_dir = run_plate_image(colab_local, model, reader, image_path)
                    st.session_state["last_plate_results"] = results
                    st.session_state["last_plate_debug_dir"] = debug_dir

        # ─── ORTAK SONUÇ GÖSTERİMİ (tekli veya toplu işleme sonrası) ───
        if st.session_state.get("last_plate_results"):
            results = st.session_state["last_plate_results"]
            debug_dir = st.session_state.get("last_plate_debug_dir")

            st.divider()
            st.subheader("Sonuçlar")
            df = pd.DataFrame([
                {
                    "Görsel": r.get("file", "-"),
                    "Plaka": r["formatted"] or "-",
                    "Güven %": f"{r['confidence']*100:.1f}",
                    "Geçerli": "✓" if r["valid"] else "✗",
                }
                for r in results
            ])
            st.dataframe(df, width='stretch')

            if debug_dir:
                for r in results:
                    stem = os.path.splitext(r.get("file", ""))[0]
                    if not stem:
                        continue
                    with st.expander(f"{r.get('file', '-')} → {r['formatted'] or 'okunamadı'}"):
                        show_plate_image_debug(debug_dir, stem)

            if st.button("Sonuçları plaka_log.csv'ye kaydet", key="plate_img_save"):
                colab_local, model, reader, gpu = load_plate_pipeline()
                colab_local.log_results_to_csv(
                    results,
                    csv_path=str(PLATE_DIR / "plaka_log.csv"),
                    overwrite=False,
                )
                st.success("plaka_log.csv güncellendi.")

        st.divider()
        st.subheader("Kayıtlı plakalar (plaka_log.csv)")
        log_path = PLATE_DIR / "plaka_log.csv"
        if log_path.exists():
            st.dataframe(pd.read_csv(log_path), width='stretch')
        else:
            st.info("Henüz kayıt yok.")

    with sub_video:
        st.subheader("Videoda gerçek zamanlı plaka takibi")
        st.caption(
            "Kare kare işlenir ve anlık olarak aşağıda gösterilir; tespit edilen "
            "plakalar hemen plaka_video_log.csv dosyasına yazılır."
        )
        source = st.radio(
            "Kaynak",
            ["Örnek video (kamera.mp4)", "Kendi videonu yükle", "Telefon kamerası (IP Webcam)"],
            key="plate_vid_source",
        )

        video_path = None
        is_live = False
        if source == "Örnek video (kamera.mp4)":
            candidate = PLATE_DIR / "kamera.mp4"
            if candidate.exists():
                video_path = str(candidate)
                st.video(video_path)
            else:
                st.warning("kamera.mp4 bulunamadı.")
        elif source == "Kendi videonu yükle":
            uploaded_v = st.file_uploader("Video yükle", type=sorted(e.strip(".") for e in VIDEO_EXTS), key="plate_vid_upload")
            if uploaded_v is not None:
                ok, msg, tmp_path = validate_video_upload(uploaded_v)
                if not ok:
                    st.error(msg)
                else:
                    video_path = tmp_path
        else:
            is_live = True
            video_path = ip_camera_input("plate_ip_url")

        if is_live:
            # Canlı yayında sabit bir süre dayatmıyoruz — sen durdurana kadar
            # sürer. Durdurmak için tarayıcının sağ üstündeki Streamlit
            # "Stop" düğmesi kullanılır.
            max_seconds = float("inf")
            st.caption("Süre sınırı yok — durdurmak için sağ üstteki **Stop** düğmesine bas.")
        else:
            max_seconds = video_duration_slider(video_path, "plate_vid_seconds")

        if video_path and st.button("İşlemeyi Başlat", key="plate_vid_run"):
            baglanti_ok = True
            if is_live:
                with st.spinner("Telefon kamerasına bağlanılıyor..."):
                    baglanti_ok = check_ip_camera(video_path)
                if not baglanti_ok:
                    st.error(
                        f"'{video_path}' adresinden görüntü alınamadı. Kontrol et: "
                        "telefonda 'Start server' basılı mı, iki cihaz aynı Wi-Fi'da mı, "
                        "adres telefon ekranındakiyle aynı mı?"
                    )
            if baglanti_ok:
                colab_local, model, reader, gpu = load_plate_pipeline()
                if model is None:
                    st.error("Model yüklenemedi (best.pt bulunamadı).")
                else:
                    frame_ph = st.empty()
                    status_ph = st.empty()
                    st.markdown("**Loglanan plakalar:**")
                    table_ph = st.empty()
                    run_plate_video(
                        video_path, model, reader, float(max_seconds),
                        frame_ph, status_ph, table_ph,
                        csv_path=str(PLATE_DIR / "plaka_video_log.csv"),
                        evidence_dir=str(PLATE_DIR / "cikti_video"),
                        live=is_live,
                    )

        st.divider()
        st.subheader("Kayıtlı plakalar (plaka_video_log.csv)")
        vlog_path = PLATE_DIR / "plaka_video_log.csv"
        if vlog_path.exists():
            st.dataframe(pd.read_csv(vlog_path), width='stretch')
        else:
            st.info("Henüz kayıt yok.")


# ───────────────────────────────── BARET & YELEK SEKMESİ ─────────────────────────────────
with tab_vest:
    render_model_metrics("vest")
    st.info(
        "VestAndPlateDetection klasöründe yalnızca eğitilmiş model dosyası (best.pt) var, "
        "örnek görsel/video bulunmuyor — kendi dosyanızı yükleyerek deneyebilirsiniz. "
        "Model sınıfları: hardhat, no-hardhat, safety-vest, no-safety-vest, person."
    )
    sub_img_v, sub_video_v = st.tabs(["Görsel", "Video (gerçek zamanlı)"])
    conf_threshold = st.slider("Tespit güven eşiği", 0.1, 0.9, 0.4, 0.05, key="vest_conf")

    with sub_img_v:
        st.subheader("Görselde baret/yelek tespiti")
        uploaded_vi = st.file_uploader("Görsel yükle", type=sorted(e.strip(".") for e in IMAGE_EXTS), key="vest_img_upload")
        image_path_v = None
        if uploaded_vi is not None:
            ok, msg = validate_image_upload(uploaded_vi)
            if not ok:
                st.error(msg)
            else:
                image_path_v = save_upload_to_temp(uploaded_vi)
                st.image(image_path_v, caption="Yüklenen görsel", width=400)

        if image_path_v and st.button("Tespit Et", key="vest_img_run"):
            vest_model = load_vest_model()
            frame = cv2.imread(image_path_v)
            with st.spinner("İşleniyor..."):
                annotated, counts = run_vest_image(vest_model, frame, conf_threshold)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Tespit sonucu", width='stretch')
            st.write(counts if counts else "Hiçbir nesne tespit edilmedi.")
            if st.button("Sonucu ihlal_log.csv'ye kaydet", key="vest_img_save"):
                log_vest_result(str(VEST_DIR / "ihlal_log.csv"), counts)
                st.success("ihlal_log.csv güncellendi.")

        st.divider()
        st.subheader("Kayıtlı ihlaller (ihlal_log.csv)")
        vest_log_path = VEST_DIR / "ihlal_log.csv"
        if vest_log_path.exists():
            st.dataframe(pd.read_csv(vest_log_path), width='stretch')
        else:
            st.info("Henüz kayıt yok.")

    with sub_video_v:
        st.subheader("Videoda gerçek zamanlı baret/yelek tespiti")
        source_v = st.radio(
            "Kaynak",
            ["Video yükle", "Telefon kamerası (IP Webcam)"],
            key="vest_vid_source",
        )

        video_path_v = None
        is_live_v = False
        if source_v == "Video yükle":
            uploaded_vv = st.file_uploader("Video yükle", type=sorted(e.strip(".") for e in VIDEO_EXTS), key="vest_vid_upload")
            if uploaded_vv is not None:
                ok, msg, tmp_path = validate_video_upload(uploaded_vv)
                if not ok:
                    st.error(msg)
                else:
                    video_path_v = tmp_path
        else:
            is_live_v = True
            video_path_v = ip_camera_input("vest_ip_url")

        if is_live_v:
            max_seconds_v = float("inf")
            st.caption("Süre sınırı yok — durdurmak için sağ üstteki **Stop** düğmesine bas.")
        else:
            max_seconds_v = video_duration_slider(video_path_v, "vest_vid_seconds")

        if video_path_v and st.button("İşlemeyi Başlat", key="vest_vid_run"):
            baglanti_ok_v = True
            if is_live_v:
                with st.spinner("Telefon kamerasına bağlanılıyor..."):
                    baglanti_ok_v = check_ip_camera(video_path_v)
                if not baglanti_ok_v:
                    st.error(
                        f"'{video_path_v}' adresinden görüntü alınamadı. Kontrol et: "
                        "telefonda 'Start server' basılı mı, iki cihaz aynı Wi-Fi'da mı, "
                        "adres telefon ekranındakiyle aynı mı?"
                    )
            if baglanti_ok_v:
                vest_model = load_vest_model()
                frame_ph_v = st.empty()
                status_ph_v = st.empty()
                st.markdown("**İhlal loglanan kareler:**")
                table_ph_v = st.empty()
                run_vest_video(
                    video_path_v, vest_model, conf_threshold, float(max_seconds_v),
                    frame_ph_v, status_ph_v, table_ph_v,
                    csv_path=str(VEST_DIR / "ihlal_log.csv"),
                    live=is_live_v,
                )


# ───────────────────────────────── YANGIN & DUMAN SEKMESİ ─────────────────────────────────
with tab_fire:
    render_model_metrics("fire")
    st.info("Ekranda ince sarı kutu = aday, kalın kırmızı/turuncu kutu = onaylı yangın/duman.")
    # Sınıf bazlı eşikler: varsayılanlar evaluate.py'nin F1-Confidence
    # eğrisinden (bkz. FireAndSmokeVideo.py DEFAULT_THRESHOLDS) veriyle
    # bulundu, sezgiyle konmadı. Adım 0.01: bu hassasiyette bulunmuş
    # değerleri (0.37 / 0.23) 0.05'lik adımla tam olarak seçemezdik.
    col_f, col_s = st.columns(2)
    fire_conf = col_f.slider("Ateş (fire) güven eşiği", 0.1, 0.9, 0.37, 0.01, key="fire_conf")
    smoke_conf = col_s.slider("Duman (smoke) güven eşiği", 0.1, 0.9, 0.23, 0.01, key="smoke_conf")

    sub_img_f, sub_video_f = st.tabs(["Görsel", "Video (gerçek zamanlı)"])

    with sub_img_f:
        st.subheader("Görselde yangın/duman tespiti")
        # Not: TTA (Test Zamanında Veri Artırma) process_photo() içinde her
        # zaman açık — burada ayrıca parametre olarak taşımaya gerek yok.
        fire_examples_dir = FIRE_DIR / "FireAndSmokeExamples"
        fire_examples_dir.mkdir(parents=True, exist_ok=True)

        source_img_f = st.radio(
            "Kaynak",
            ["Örnek klasördeki görselleri kullan (FireAndSmokeExamples/)", "Kendi görselini yükle"],
            key="fire_img_source",
        )

        if source_img_f == "Örnek klasördeki görselleri kullan (FireAndSmokeExamples/)":
            # ─── Klasöre yeni görsel ekleme ───
            # Bir klasördeki tüm fotoğrafları toplu gezme özelliği; ek olarak
            # o klasöre doğrudan arayüzden fotoğraf da eklenebiliyor.
            new_examples = st.file_uploader(
                "FireAndSmokeExamples/ klasörüne yeni görsel ekle (birden fazla seçilebilir)",
                type=sorted(e.strip(".") for e in IMAGE_EXTS),
                accept_multiple_files=True,
                key="fire_examples_add",
            )
            if new_examples:
                eklenen = 0
                for nf in new_examples:
                    ok, msg = validate_image_upload(nf)
                    if ok:
                        (fire_examples_dir / nf.name).write_bytes(nf.getvalue())
                        eklenen += 1
                    else:
                        st.error(f"{nf.name}: {msg}")
                if eklenen:
                    st.success(f"{eklenen} görsel FireAndSmokeExamples/ klasörüne eklendi.")
                    st.session_state.pop("fire_batch_results", None)  # eski sonuçlar artık gecerli degil
                    st.rerun()  # galeri listesi yeni eklenenleri hemen göstersin

            samples_f = sorted(p for p in fire_examples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not samples_f:
                st.warning(f"'{fire_examples_dir}' klasöründe örnek görsel bulunamadı. Yukarıdan ekleyebilirsin.")
            else:
                st.caption(f"'{fire_examples_dir}' klasöründeki {len(samples_f)} görsel:")
                preview_cols_f = st.columns(6)
                for i, p in enumerate(samples_f):
                    preview_cols_f[i % 6].image(str(p), caption=p.name, width='stretch')

                if st.button(f"Tüm örnek görselleri işle ({len(samples_f)} adet)", key="fire_batch_run"):
                    from FireAndSmokeVideo import process_photo
                    fire_model = load_fire_model()
                    batch_results_f = []
                    with st.spinner(f"{len(samples_f)} görsel işleniyor..."):
                        for p in samples_f:
                            frame_b = cv2.imread(str(p))
                            annotated_b, counts_b = process_photo(
                                fire_model, frame_b,
                                thresholds={"fire": fire_conf, "smoke": smoke_conf},
                            )
                            batch_results_f.append((p.name, annotated_b, counts_b))
                    st.session_state["fire_batch_results"] = batch_results_f

            if st.session_state.get("fire_batch_results"):
                st.divider()
                st.subheader("Toplu işleme sonuçları")
                for name, annotated_b, counts_b in st.session_state["fire_batch_results"]:
                    ozet_b = counts_b if counts_b else "tespit yok"
                    with st.expander(f"{name} — {ozet_b}"):
                        st.image(cv2.cvtColor(annotated_b, cv2.COLOR_BGR2RGB), width='stretch')
        else:
            uploaded_fi = st.file_uploader("Görsel yükle", type=sorted(e.strip(".") for e in IMAGE_EXTS), key="fire_img_upload")
            image_path_f = None
            if uploaded_fi is not None:
                ok, msg = validate_image_upload(uploaded_fi)
                if not ok:
                    st.error(msg)
                else:
                    image_path_f = save_upload_to_temp(uploaded_fi)
                    st.image(image_path_f, caption="Yüklenen görsel", width=400)

            if image_path_f and st.button("Tespit Et", key="fire_img_run"):
                from FireAndSmokeVideo import process_photo
                fire_model = load_fire_model()
                frame_f = cv2.imread(image_path_f)
                with st.spinner("İşleniyor..."):
                    annotated_f, counts_f = process_photo(
                        fire_model, frame_f,
                        thresholds={"fire": fire_conf, "smoke": smoke_conf},
                    )
                st.image(cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB), caption="Tespit sonucu", width='stretch')
                st.write(counts_f if counts_f else "Yangın/duman tespit edilmedi.")

    with sub_video_f:
        st.subheader("Videoda gerçek zamanlı yangın/duman izleme")
        source_f = st.radio(
            "Kaynak",
            ["Video yükle", "Telefon kamerası (IP Webcam)"],
            key="fire_vid_source",
        )

        video_path_f = None
        is_live_f = False
        if source_f == "Video yükle":
            uploaded_fv = st.file_uploader("Video yükle", type=sorted(e.strip(".") for e in VIDEO_EXTS), key="fire_vid_upload")
            if uploaded_fv is not None:
                ok, msg, tmp_path = validate_video_upload(uploaded_fv)
                if not ok:
                    st.error(msg)
                else:
                    video_path_f = tmp_path
        else:
            is_live_f = True
            video_path_f = ip_camera_input("fire_ip_url")

        if is_live_f:
            max_seconds_f = float("inf")
            st.caption("Süre sınırı yok — durdurmak için sağ üstteki **Stop** düğmesine bas.")
        else:
            max_seconds_f = video_duration_slider(video_path_f, "fire_vid_seconds")

        if video_path_f and st.button("İzlemeyi Başlat", key="fire_vid_run"):
            baglanti_ok_f = True
            if is_live_f:
                with st.spinner("Telefon kamerasına bağlanılıyor..."):
                    baglanti_ok_f = check_ip_camera(video_path_f)
                if not baglanti_ok_f:
                    st.error(
                        f"'{video_path_f}' adresinden görüntü alınamadı. Kontrol et: "
                        "telefonda 'Start server' basılı mı, iki cihaz aynı Wi-Fi'da mı, "
                        "adres telefon ekranındakiyle aynı mı?"
                    )
            if baglanti_ok_f:
                from FireAndSmokeVideo import FireSmokeMonitor
                fire_model = load_fire_model()
                # Her izleme oturumu İÇİN YENİ izleyici: iz geçmişi (N-of-M
                # pencereleri) temiz başlar; model ise cache'ten paylaşılır.
                izleyici = FireSmokeMonitor(
                    model=fire_model,
                    thresholds={"fire": fire_conf, "smoke": smoke_conf},
                    record_dir=str(FIRE_DIR / "YanginKayitlari"),
                    csv_path=str(FIRE_DIR / "yangin_olay_log.csv"),
                )
                frame_ph_f = st.empty()
                status_ph_f = st.empty()
                st.markdown("**Onaylanan yangın/duman olayları:**")
                table_ph_f = st.empty()
                run_fire_video(
                    video_path_f, izleyici, float(max_seconds_f),
                    frame_ph_f, status_ph_f, table_ph_f,
                    csv_path=str(FIRE_DIR / "yangin_olay_log.csv"),
                    live=is_live_f,
                )

        st.divider()
        st.subheader("Olay kayıtları (yangin_olay_log.csv)")
        fire_log_path = FIRE_DIR / "yangin_olay_log.csv"
        if fire_log_path.exists():
            st.dataframe(pd.read_csv(fire_log_path), width='stretch')
        else:
            st.info("Henüz kayıt yok.")

        # Son kanıt fotoğrafları: onaylanan olayların anotasyonlu kareleri
        kayit_dir_f = FIRE_DIR / "YanginKayitlari"
        if kayit_dir_f.exists():
            son_kanitlar = sorted(kayit_dir_f.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[-6:]
            if son_kanitlar:
                st.subheader("Son kanıt fotoğrafları")
                cols_k = st.columns(3)
                for i, p in enumerate(reversed(son_kanitlar)):
                    cols_k[i % 3].image(str(p), caption=p.name, width='stretch')


# ───────────────────────────────── TIBBİ PPE SEKMESİ ─────────────────────────────────
with tab_med:
    render_model_metrics("medical")
    med_weights = medical_weights_or_none()

    if med_weights is None:
        # Model henüz eğitimde: sekme hazır ama pasif. best.pt,
        # MedicalPPE/ klasörüne kopyalandığı anda (herhangi bir etkileşimle
        # gelen ilk rerun'da) aşağıdaki arayüz kendiliğinden açılır.
        st.warning(
            "Tıbbi PPE modeli henüz eğitimde. Eğitim bitince **best.pt** dosyasını "
            "`MedicalPPE/` klasörüne kopyala — bu sekme kendiliğinden aktifleşecek. "
            "Sonrasında `MedicalPPE/evaluate.py` ile sınıf bazlı eşikleri kalibre "
            "etmeyi unutma (detay: MedicalPPE/README.md)."
        )
        st.button("Modeli tekrar ara", key="med_retry")
    else:
        st.info(
            "İnce yeşil kutu = takılı ekipman, sarı = aday ihlal, kalın kırmızı = onaylı ihlal. "
            "Denetlenen ekipmanlar modelin sınıflarından otomatik çıkarılır (eldiven, bone, önlük, gözlük...)."
        )
        med_model = load_medical_model(med_weights)
        med_conf = st.slider(
            "Tespit güven eşiği", 0.1, 0.9, 0.4, 0.05, key="med_conf",
            help="Geçici genel eşik. evaluate.py çalıştırıp sınıf bazlı eşikleri "
                 "MedicalPPEVideo.py'deki DEFAULT_THRESHOLDS'a yazınca oradaki "
                 "değerler sınıf bazında önceliklidir.",
        )

        sub_img_m, sub_video_m = st.tabs(["Görsel", "Video (gerçek zamanlı)"])

        with sub_img_m:
            st.subheader("Görselde tıbbi PPE denetimi")
            med_examples_dir = MED_DIR / "MedicalPPEExamples"
            med_examples_dir.mkdir(parents=True, exist_ok=True)

            source_img_m = st.radio(
                "Kaynak",
                ["Örnek klasördeki görselleri kullan (MedicalPPEExamples/)", "Kendi görselini yükle"],
                key="med_img_source",
            )

            if source_img_m == "Örnek klasördeki görselleri kullan (MedicalPPEExamples/)":
                new_examples_m = st.file_uploader(
                    "MedicalPPEExamples/ klasörüne yeni görsel ekle (birden fazla seçilebilir)",
                    type=sorted(e.strip(".") for e in IMAGE_EXTS),
                    accept_multiple_files=True,
                    key="med_examples_add",
                )
                if new_examples_m:
                    eklenen_m = 0
                    for nf in new_examples_m:
                        ok, msg = validate_image_upload(nf)
                        if ok:
                            (med_examples_dir / nf.name).write_bytes(nf.getvalue())
                            eklenen_m += 1
                        else:
                            st.error(f"{nf.name}: {msg}")
                    if eklenen_m:
                        st.success(f"{eklenen_m} görsel MedicalPPEExamples/ klasörüne eklendi.")
                        st.session_state.pop("med_batch_results", None)
                        st.rerun()

                samples_m = sorted(p for p in med_examples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                if not samples_m:
                    st.warning(f"'{med_examples_dir}' klasöründe örnek görsel bulunamadı. Yukarıdan ekleyebilirsin.")
                else:
                    st.caption(f"'{med_examples_dir}' klasöründeki {len(samples_m)} görsel:")
                    preview_cols_m = st.columns(6)
                    for i, p in enumerate(samples_m):
                        preview_cols_m[i % 6].image(str(p), caption=p.name, width='stretch')

                    if st.button(f"Tüm örnek görselleri işle ({len(samples_m)} adet)", key="med_batch_run"):
                        from MedicalPPEVideo import process_photo as med_process_photo
                        batch_results_m = []
                        with st.spinner(f"{len(samples_m)} görsel işleniyor..."):
                            for p in samples_m:
                                frame_bm = cv2.imread(str(p))
                                annotated_bm, counts_bm, viol_bm = med_process_photo(
                                    med_model, frame_bm, default_conf=med_conf,
                                )
                                batch_results_m.append((p.name, annotated_bm, counts_bm, viol_bm))
                        st.session_state["med_batch_results"] = batch_results_m

                if st.session_state.get("med_batch_results"):
                    st.divider()
                    st.subheader("Toplu işleme sonuçları")
                    for name, annotated_bm, counts_bm, viol_bm in st.session_state["med_batch_results"]:
                        ozet_m = "; ".join(viol_bm) if viol_bm else (counts_bm if counts_bm else "tespit yok")
                        with st.expander(f"{name} — {ozet_m}"):
                            st.image(cv2.cvtColor(annotated_bm, cv2.COLOR_BGR2RGB), width='stretch')
                            if viol_bm:
                                st.error(" | ".join(viol_bm))
            else:
                uploaded_mi = st.file_uploader("Görsel yükle", type=sorted(e.strip(".") for e in IMAGE_EXTS), key="med_img_upload")
                image_path_m = None
                if uploaded_mi is not None:
                    ok, msg = validate_image_upload(uploaded_mi)
                    if not ok:
                        st.error(msg)
                    else:
                        image_path_m = save_upload_to_temp(uploaded_mi)
                        st.image(image_path_m, caption="Yüklenen görsel", width=400)

                if image_path_m and st.button("Denetle", key="med_img_run"):
                    from MedicalPPEVideo import process_photo as med_process_photo
                    frame_m = cv2.imread(image_path_m)
                    with st.spinner("İşleniyor..."):
                        annotated_m, counts_m, viol_m = med_process_photo(
                            med_model, frame_m, default_conf=med_conf,
                        )
                    st.image(cv2.cvtColor(annotated_m, cv2.COLOR_BGR2RGB), caption="Denetim sonucu", width='stretch')
                    st.write(counts_m if counts_m else "Hiçbir nesne tespit edilmedi.")
                    if viol_m:
                        st.error(" | ".join(viol_m))
                    elif counts_m:
                        st.success("Anlık ihlal görünmüyor.")

        with sub_video_m:
            st.subheader("Videoda gerçek zamanlı tıbbi PPE izleme")
            source_m = st.radio(
                "Kaynak",
                ["Video yükle", "Telefon kamerası (IP Webcam)"],
                key="med_vid_source",
            )

            video_path_m = None
            is_live_m = False
            if source_m == "Video yükle":
                uploaded_mv = st.file_uploader("Video yükle", type=sorted(e.strip(".") for e in VIDEO_EXTS), key="med_vid_upload")
                if uploaded_mv is not None:
                    ok, msg, tmp_path = validate_video_upload(uploaded_mv)
                    if not ok:
                        st.error(msg)
                    else:
                        video_path_m = tmp_path
            else:
                is_live_m = True
                video_path_m = ip_camera_input("med_ip_url")

            if is_live_m:
                max_seconds_m = float("inf")
                st.caption("Süre sınırı yok — durdurmak için sağ üstteki **Stop** düğmesine bas.")
            else:
                max_seconds_m = video_duration_slider(video_path_m, "med_vid_seconds")

            if video_path_m and st.button("İzlemeyi Başlat", key="med_vid_run"):
                baglanti_ok_m = True
                if is_live_m:
                    with st.spinner("Telefon kamerasına bağlanılıyor..."):
                        baglanti_ok_m = check_ip_camera(video_path_m)
                    if not baglanti_ok_m:
                        st.error(
                            f"'{video_path_m}' adresinden görüntü alınamadı. Kontrol et: "
                            "telefonda 'Start server' basılı mı, iki cihaz aynı Wi-Fi'da mı, "
                            "adres telefon ekranındakiyle aynı mı?"
                        )
                if baglanti_ok_m:
                    from MedicalPPEVideo import MedicalPPEMonitor
                    # Her izleme oturumu İÇİN YENİ izleyici: iz geçmişi (N-of-M
                    # pencereleri) temiz başlar; model ise cache'ten paylaşılır.
                    med_izleyici = MedicalPPEMonitor(
                        model=med_model,
                        default_conf=med_conf,
                        record_dir=str(MED_DIR / "IhlalKayitlari"),
                        csv_path=str(MED_DIR / "medikal_ihlal_log.csv"),
                    )
                    st.caption(
                        f"İzleme modu: **{med_izleyici.mode}** — denetlenen ekipman: "
                        f"{', '.join(sorted(med_izleyici.required_items)) or '-'}"
                    )
                    if med_izleyici.mode == "presence":
                        st.warning(
                            "Modelde ne kişi ne de no_* sınıfı var; ihlal çıkarımı yapılamaz, "
                            "yalnızca tespitler gösterilir (detay: MedicalPPE/README.md)."
                        )
                    frame_ph_m = st.empty()
                    status_ph_m = st.empty()
                    st.markdown("**Onaylanan PPE ihlalleri:**")
                    table_ph_m = st.empty()
                    run_medical_video(
                        video_path_m, med_izleyici, float(max_seconds_m),
                        frame_ph_m, status_ph_m, table_ph_m,
                        csv_path=str(MED_DIR / "medikal_ihlal_log.csv"),
                        live=is_live_m,
                    )

        st.divider()
        st.subheader("İhlal kayıtları (medikal_ihlal_log.csv)")
        med_log_path = MED_DIR / "medikal_ihlal_log.csv"
        if med_log_path.exists():
            st.dataframe(pd.read_csv(med_log_path), width='stretch')
        else:
            st.info("Henüz kayıt yok.")

        # Son kanıt fotoğrafları: onaylanan ihlallerin anotasyonlu kareleri
        kayit_dir_m = MED_DIR / "IhlalKayitlari"
        if kayit_dir_m.exists():
            son_kanitlar_m = sorted(kayit_dir_m.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[-6:]
            if son_kanitlar_m:
                st.subheader("Son kanıt fotoğrafları")
                cols_km = st.columns(3)
                for i, p in enumerate(reversed(son_kanitlar_m)):
                    cols_km[i % 3].image(str(p), caption=p.name, width='stretch')
