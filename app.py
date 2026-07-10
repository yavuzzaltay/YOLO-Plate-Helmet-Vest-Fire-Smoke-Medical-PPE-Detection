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
except Exception:
    PLATE_DIR = Path("PlateDetection")
    VEST_DIR = Path("VestAndPlateDetection")

sys.path.insert(0, str(PLATE_DIR))
sys.path.insert(0, str(VEST_DIR))

st.set_page_config(page_title="İyex Tespit Demo", layout="wide")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
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


# ═══════════════════════════════════════ ARAYÜZ ═══════════════════════════════════════

st.title("İyex Tespit Demo")
st.caption("Plaka okuma ve baret/yelek tespiti modellerini tarayıcıdan deneyin.")

tab_plate, tab_vest = st.tabs(["🚗 Plaka Tespiti", "🦺 Baret & Yelek Tespiti"])

# ───────────────────────────────── PLAKA TESPİTİ SEKMESİ ─────────────────────────────────
with tab_plate:
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
