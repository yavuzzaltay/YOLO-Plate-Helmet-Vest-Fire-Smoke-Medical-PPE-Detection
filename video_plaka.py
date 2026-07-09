# -*- coding: utf-8 -*-
"""
=============================================================================
VİDEO PLAKA TESPİTİ — ÖRNEKLENMİŞ TESPİT + İZ TAKİBİ + İZ SONU OYLAMASI
=============================================================================

NEDEN HER KAREYİ İŞLEMİYORUZ?
60fps videoda her kareyi işlemek hem gereksiz (ardışık kareler neredeyse
aynıdır) hem de imkânsız derecede yavaştır (kare başına 10 OCR çağrısı).
NEDEN TEK KARE DE YETMEZ? Tek karede plaka gölgede/bulanık/açılı
yakalanabilir; video ise aynı plakayı onlarca farklı karede sunar —
bu çoklu kanıt tek karenin çözemediği karakter hatalarını çözer.

MİMARİ (3 KATMAN):
1. ÖRNEKLEME: Her FRAME_SKIP karede bir YOLO tespiti (60fps'te 6 kare
   atlama = saniyede 10 tespit; plaka en az ~1sn görünür, kaçmaz).
2. İZ TAKİBİ: Ardışık tespit kutuları IoU ile eşleştirilir → her fiziksel
   plaka bir "iz" (track) olur. Ağır OCR her karede DEĞİL, iz boyunca
   biriken kırpımlardan yalnızca EN İYİ birkaçında çalışır. Kırpım
   kalitesi = genişlik × netlik (Laplace varyansı): büyük ve keskin
   kırpım OCR için en değerlisidir.
3. ARTIMLI OKUMA + KONSENSÜS ERKEN ÇIKIŞI  [DEĞİŞTİ]
   NEDEN? Eskiden OCR yalnızca iz KAPANINCA (araç kadrajdan çıkınca)
   çalışıyordu ("batch" mantık). Bu, tam önümüzde uzun süre aynı hızda
   giden ve hiç kaybolmayan bir aracın plakasını SONSUZA DEK işlemeden
   bırakıyordu — ekranda kutu görünüyor ama sonuç asla çıkmıyordu.
   Gerçek ANPR sistemleri (OpenALPR, bariyer kontrol sistemleri vb.)
   bunu böyle yapmaz: bariyer açma gibi kullanımlarda araç kadrajdan
   çıkmadan ÖNCE karara varmak gerekir. Standart yaklaşım ARTIMLI
   KONSENSÜS'tür: iz yaşarken belirli aralıklarla (yeterli yeni kırpım
   biriktikçe) OCR tekrar denenir; ≥2 bağımsız kırpım aynı sonuca
   varırsa (veya tek okuma ≥0.80 güvenliyse) sonuç HEMEN kesinleşir ve
   CSV'ye yazılır — izin kapanması beklenmez. İz kapanması artık sadece
   bu eşiğe hiç ulaşamayan zayıf izler için bir "son çare" denemesidir.

Kullanım:
    python video_plaka.py                    # varsayılan video, izleme penceresi AÇIK
    python video_plaka.py yol/video.mp4      # başka video
    python video_plaka.py yol/video.mp4 90   # yalnız ilk 90 saniyeyi işle
    python video_plaka.py yol/video.mp4 sessiz  # pencere olmadan (sunucu modu)

İzleme penceresinde: yeşil kutu = aktif iz, alt şerit = loglanan plakalar.
Çıkmak için Q veya ESC (o ana kadarki kayıtlar CSV'de kalır).
"""

import os
import sys
import csv
import time
from datetime import datetime

import cv2
import torch

# Görüntü hattının tüm ağır işleri colab_local'dan gelir — video katmanı
# yalnızca örnekleme/takip/oylama düzenini kurar (tek kaynak ilkesi)
from colab_local import (
    initialize_model,
    initialize_ocr,
    process_plate_candidate,
    _character_vote,
    format_turkish_plate_ex,
)

# ─── AYARLAR ───
FRAME_SKIP = 6        # 60fps'te saniyede ~10 tespit
YOLO_CONF = 0.15      # izleme bağlamında düşük eşik güvenli (iz + OCR eler)
IOU_MATCH = 0.30      # kutu → iz eşleştirme eşiği
MISS_LIMIT = 15       # iz bu kadar örnek güncellenmezse kapanır (~1.5sn)
BEST_CROPS = 5        # iz başına tam OCR yapılacak en iyi kırpım sayısı
                      # (3'te tek kırpımlık eksik okuma '34 JH 442'nin
                      # doğru '34 JH 4438'i yendiği görüldü — daha çok
                      # bağımsız kırpım = daha sağlam oylama)
KEEP_CROPS = 8        # iz boyunca bellekte tutulan en iyi kırpım sayısı
CROP_GAP = 0.4        # tutulan kırpımlar arası asgari zaman farkı (sn).
                      # NEDEN? Kalite sıralaması ardışık (neredeyse özdeş)
                      # kareleri seçiyordu → "3 kırpım" aslında tek anın
                      # 3 kopyasıydı, hataları da ortaktı. Zamana yayılmış
                      # kırpımlar BAĞIMSIZ kanıttır; oylama ancak öyle işler.
CROP_PAD = 0.45       # kırpım kenar payı (plaka yüksekliğinin oranı)
MIN_BOX_W = 60        # bundan dar plaka kırpımı OCR için umutsuz, saklanmaz
DEDUP_WINDOW = 120.0  # aynı plaka bu süre (video sn) içinde tekrar loglanmaz
                      # (dashcam'de öndeki araç dakikalarca takip edilir;
                      # her tekrar görünüş yeni kayıt DEĞİLDİR)
CSV_PATH = "plaka_video_log.csv"


def pick_imgsz(frame_width):
    """
    YOLO giriş çözünürlüğü kare genişliğine uyarlanır:
    4K karede 1280'e küçültmek plakaları 3'te 1'ine indirip uzak
    plakaları kaybettirir; 1920 tespit penceresi 4K'nın getirdiği
    ekstra plaka pikselini korur. 1080p ve altında 1280 yeterli
    (deneylerle doğrulanmış davranış).
    """
    return 1920 if frame_width >= 3000 else 1280

# Plaka geometri filtresi (colab_local.detect_plate_boxes ile aynı sınırlar)
PLATE_AR_MIN, PLATE_AR_MAX = 1.5, 7.5
MIN_WIDTH, MIN_HEIGHT = 30, 10


def _iou(a, b):
    """İki (x1,y1,x2,y2) kutusunun kesişim/birleşim oranı."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _crop_quality(gray_crop):
    """
    Kırpım kalitesi: genişlik × netlik.
    Netlik = Laplace varyansı (bulanık görüntüde kenarlar yumuşar,
    ikinci türev varyansı düşer). Genişlik çarpanı: aynı netlikte
    daha büyük plaka görüntüsü her zaman daha çok karakter detayı taşır.
    """
    lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
    return gray_crop.shape[1] * min(lap_var, 2000.0)


class PlateTrack:
    """Tek bir fiziksel plakanın video boyunca izi."""
    _next_id = 1

    def __init__(self, box, sample_idx, video_sec):
        self.id = PlateTrack._next_id
        PlateTrack._next_id += 1
        self.box = box                    # son bilinen (x1,y1,x2,y2)
        self.last_seen = sample_idx       # son eşleşen örnek numarası
        self.first_sec = video_sec        # ilk görülme (video saniyesi)
        self.last_sec = video_sec
        self.hits = 1                     # kaç örnekte görüldü
        self.crops = []                   # [(kalite, altkare, göreli_kutu, conf)]
        self.resolved = False             # çapraz doğrulanmış sonuç bulundu mu?
        self.resolved_text = None         # bulunduysa metni (canlı overlay için)
        self.last_attempt_count = 0       # son OCR denemesindeki kırpım sayısı

    def update(self, box, sample_idx, video_sec):
        self.box = box
        self.last_seen = sample_idx
        self.last_sec = video_sec
        self.hits += 1

    def add_crop(self, frame, box, conf, video_sec):
        """Kutu çevresinden paylı altkare kes, kaliteliyse sakla."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_W:
            return
        pad = int(bh * CROP_PAD)
        H, W = frame.shape[:2]
        sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
        sx2, sy2 = min(W, x2 + pad), min(H, y2 + pad)
        sub = frame[sy1:sy2, sx1:sx2].copy()
        rel_box = (x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1)

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        quality = _crop_quality(gray)

        # ─── ZAMANSAL ÇEŞİTLİLİK ───
        # CROP_GAP içinde zaten bir kırpım varsa ikisinden yalnızca iyi
        # olan kalır — böylece tutulan kırpımlar izin ömrüne yayılır ve
        # oylamaya BAĞIMSIZ kanıt taşır (ardışık kare kopyaları değil)
        for i, (q, _, _, _, sec) in enumerate(self.crops):
            if abs(sec - video_sec) < CROP_GAP:
                if quality > q:
                    self.crops[i] = (quality, sub, rel_box, conf, video_sec)
                return

        self.crops.append((quality, sub, rel_box, conf, video_sec))
        # Bellek kontrolü: yalnızca en iyi KEEP_CROPS kırpım tutulur
        self.crops.sort(key=lambda c: c[0], reverse=True)
        del self.crops[KEEP_CROPS:]


def detect_boxes_fast(model, frame, device, imgsz):
    """
    Video için TEK geçişli YOLO tespiti + geometri filtresi.
    Fotoğraf hattındaki 3 geçişli detect_plate_boxes video karesi başına
    fazla pahalı; izleme zaten zamansal fazlalık sağladığından tek geçiş
    yeterli (bir karede kaçan plaka sonraki örneklerde yakalanır).
    """
    results = model.predict(source=frame, device=device, verbose=False,
                            imgsz=imgsz, conf=YOLO_CONF)
    boxes = results[0].boxes
    out = []
    if boxes is None:
        return out
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        ar = bw / bh
        if (PLATE_AR_MIN <= ar <= PLATE_AR_MAX
                and bw >= MIN_WIDTH and bh >= MIN_HEIGHT):
            out.append((x1, y1, x2, y2, float(boxes[i].conf[0])))
    return out


def attempt_read(track, reader, output_dir, context_label):
    """
    İzin O ANA KADAR biriken en iyi kırpımlarını tam OCR hattından
    geçirir, karakter oylamasıyla birleştirir ve çapraz doğrulama
    sonucunu döner. CSV'ye YAZMAZ — hem iz canlıyken artımlı kontrol
    hem de iz kapanışı için ortak kullanılan salt-okuma denemesidir.
    """
    if not track.crops:
        return None

    # ─── KANIT EŞİĞİ ───
    # 1-2 örneklik izler neredeyse her zaman yanlış tespittir (tabela,
    # reklam yazısı, tek karelik parlamalar) — duman testinde bu izlerin
    # tam OCR sonuçları hep boş/çöp çıktı. Gerçek bir plaka 10Hz
    # örneklemede en az ~0.3sn (3 örnek) izlenir. İstisna: büyük ve net
    # tek kırpım (genişlik ≥120px) yine de denenir (çok hızlı geçen araç).
    best_rel = track.crops[0][2]  # en iyi kırpımın plaka kutusu
    if track.hits < 3 and (best_rel[2] - best_rel[0]) < 120:
        return None

    print(f"\n{'─' * 60}")
    print(f"[İZ #{track.id}] {context_label} "
          f"({track.hits} örnek, {len(track.crops)} kırpım) → tam OCR")

    readings = []  # (formatted, conf, valid)
    for k, (quality, sub, rel_box, dconf, _sec) in enumerate(track.crops[:BEST_CROPS], 1):
        x1, y1, x2, y2 = rel_box
        ar = (x2 - x1) / max(1, y2 - y1)
        box6 = (x1, y1, x2, y2, dconf, ar)
        r = process_plate_candidate(sub, box6, reader, output_dir,
                                    f"iz{track.id}", save_debug=False,
                                    tag=f"_k{k}")
        readings.append((r['formatted'], r['confidence'], r['valid']))

    # ─── İZ İÇİ OYLAMA ───
    # Geçerli okumalar arasında karakter oylaması; ayrıca aynı sonucun
    # birden çok BAĞIMSIZ karede tekrarı en güçlü kanıttır
    valid_pool = [(f, c) for f, c, v in readings if v]
    final_text, final_conf, final_valid = "", 0.0, False

    if valid_pool:
        voted_text, voted_conf = _character_vote(valid_pool)
        # Oylama çıktısı ham metindir (boşluksuz olabilir) — CSV ve
        # tekrar-önleme anahtarı tutarlılığı için yeniden formatlanır
        if voted_text:
            voted_fmt, voted_valid, _ = format_turkish_plate_ex(voted_text)
            if voted_valid:
                voted_text = voted_fmt
            else:
                voted_text = None  # oylama bozuk sonuç ürettiyse güvenme
        if voted_text:
            final_text, final_conf, final_valid = voted_text, voted_conf, True
        else:
            valid_pool.sort(key=lambda p: p[1], reverse=True)
            final_text, final_conf, final_valid = (valid_pool[0][0],
                                                   valid_pool[0][1], True)
        # ─── KARELER ARASI KONSENSÜS ───  [GÜÇLENDİRİLDİ]
        # Aynı plakanın birden çok ZAMANSAL BAĞIMSIZ karede birebir aynı
        # okunması tek karedeki güven yüzdesinden güçlü kanıttır — ama
        # kademeli:
        #   ≥3 kare aynı → güven tabanı 0.60 (kalite kapısı açılır).
        #     Üç ayrı anın aynı 7-8 karakteri tesadüfen üretmesi ihmal
        #     edilebilir.
        #   =2 kare aynı → yalnızca gerçek güvenlerin en yükseği alınır,
        #     TABAN YOK. NEDEN? Gerçek hata: '35 FR 056' iki karede aynı
        #     sistematik hatayla '35 G 0056' okundu (aynı plaka, aynı
        #     font, aynı bozulma → hatalar korelasyonlu) ve 0.60 tabanı
        #     bu yanlışı CSV'ye taşıdı. İki kare, sistematik hatayı
        #     dışlamak için yetersiz; üç kare istatistiksel olarak yeterli.
        agree_confs = [c for f, c in valid_pool if f == final_text]
        if len(agree_confs) >= 3:
            final_conf = max(final_conf, max(agree_confs), 0.60)
            print(f"[İZ #{track.id}] Konsensüs: '{final_text}' "
                  f"{len(agree_confs)} bağımsız karede aynı → güven "
                  f"%{final_conf * 100:.0f}'e yükseltildi")
        elif len(agree_confs) == 2:
            final_conf = max(final_conf, max(agree_confs))
    elif readings:
        # Hiç geçerli okuma yok → en güvenli ham sonucu raporla (loglanmaz)
        readings.sort(key=lambda r: r[1], reverse=True)
        final_text, final_conf = readings[0][0], readings[0][1]

    status = "GEÇERLİ" if final_valid else "geçersiz"
    print(f"[İZ #{track.id}] Okuma sonucu: '{final_text}' "
          f"({status}, güven: {final_conf:.2f})")

    # ─── ÇAPRAZ DOĞRULAMA ŞARTI ───
    # Sonuç için iki bağımsız kanıt yolundan biri şart:
    #   (a) ≥2 kırpım birebir aynı okumada mutabık (kareler arası teyit)
    #   (b) tek okuma ama güveni ≥ 0.80 (kendi başına yeterince net)
    # NEDEN? Gerçek hatalar: tek kırpımlık '37 ER 491' (güven 0.72,
    # gerçek plaka '34 TGR 49') ve mutabakatsız '34 FYR 15' (0.64,
    # gerçek '34 FYR 150') kalite kapısını geçip CSV'yi kirletti.
    # Ne mutabakatı ne yüksek güveni olan okuma İDDİA EDİLMEZ.
    agree_count = len([c for f, c in valid_pool if f == final_text]) if final_valid else 0
    cross_validated = agree_count >= 2 or final_conf >= 0.80
    if final_valid and not cross_validated:
        print(f"[İZ #{track.id}] Çapraz doğrulama YOK (mutabakat: "
              f"{agree_count} kırpım, güven: {final_conf:.2f} < 0.80)")

    return {'text': final_text, 'conf': final_conf, 'valid': final_valid,
            'cross_validated': cross_validated}


def log_result(track, result, logged, csv_writer, csv_file, output_dir):
    """
    Çapraz doğrulanmış bir okumayı kalite kapısından geçirip CSV'ye
    yazar (yakın-eşleşme tekrar önlemesiyle). `attempt_read`'den ayrı
    tutulur çünkü artımlı kontrol sırasında birden çok deneme yapılabilir
    ama YAZMA işlemi yalnızca çapraz doğrulama sağlandığı an, bir kez
    gerçekleşmeli.
    """
    if not result:
        return None
    final_text, final_conf = result['text'], result['conf']
    if not (result['valid'] and result['cross_validated'] and final_conf >= 0.35):
        return None

    # Tekrar kontrolü YAKIN EŞLEŞME ile yapılır: birebir aynı metin
    # VEYA biri diğerinin altdizisi olan tek karakter farkı.
    # NEDEN? Gerçek hata: aynı plaka bir izde '35 FR 056', sonraki
    # izde çerçeve kenarından hayalet '1' ile '35 FR 0561' okundu ve
    # iki ayrı kayıt oluştu. Hayalet ekleme/düşme tek karakterliktir;
    # gerçekte farklı iki plaka tek karakter altdizi farkıyla bu
    # pencere içinde görülürse kaybedilecek bilgi ihmal edilebilir.
    def _near(a, b):
        a, b = a.replace(" ", ""), b.replace(" ", "")
        if a == b:
            return True
        if len(a) == len(b):
            # Eş uzunlukta TEK karakter farkı da aynı plakadır
            # (gerçek hata: '35 FR 056' ikinci izde 3→0 hatasıyla
            # '05 FR 056' okundu ve ayrı kayıt açıldı)
            return sum(1 for x, y in zip(a, b) if x != y) <= 1
        if abs(len(a) - len(b)) == 1:
            short, long = (a, b) if len(a) < len(b) else (b, a)
            it = iter(long)
            return all(ch in it for ch in short)
        return False

    dup_key = next((k for k, t in logged.items()
                    if (track.first_sec - t) < DEDUP_WINDOW
                    and _near(final_text, k)), None)
    if dup_key is not None:
        logged[dup_key] = track.last_sec  # pencereyi tazele: araç
        # hâlâ takipte olduğu sürece tekrar kayıt açılmasın
        print(f"[İZ #{track.id}] '{final_text}' ≈ '{dup_key}' "
              f"({DEDUP_WINDOW:.0f}sn penceresinde) → tekrar yazılmadı")
        return None

    logged[final_text] = track.last_sec
    csv_writer.writerow([
        final_text,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{final_conf * 100:.1f}",
        f"{track.first_sec:.1f}",
    ])
    csv_file.flush()  # video işleme uzun sürer; anında diske yaz
    print(f"[İZ #{track.id}] ✓ CSV'ye yazıldı: {final_text} "
          f"(%{final_conf * 100:.1f}, video {track.first_sec:.1f}s)")
    # En iyi kırpımı kanıt olarak kaydet
    if output_dir:
        cv2.imwrite(os.path.join(
            output_dir,
            f"iz{track.id}_{final_text.replace(' ', '')}.jpg"),
            track.crops[0][1])
    return final_text  # yalnızca GERÇEKTEN yazılan kayıt sayılır


def _draw_overlay(frame, tracks, recent_plates, video_sec, realtime_ratio):
    """İzleme penceresi için kutu + bilgi şeridi çizer (1280px'e ölçekli)."""
    H, W = frame.shape[:2]
    scale = 1280.0 / W
    view = cv2.resize(frame, (1280, int(H * scale)))

    for tr in tracks:
        x1, y1, x2, y2 = (int(v * scale) for v in tr.box)
        # Çözülmüş izler farklı renkte (camgöbeği) — okuma bulundu demek;
        # yeşil kutu hâlâ "okunmaya çalışılıyor" durumunu gösterir
        color = (255, 220, 0) if tr.resolved_text else (0, 220, 0)
        label = f"iz#{tr.id}: {tr.resolved_text}" if tr.resolved_text else f"iz#{tr.id}"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.putText(view, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Alt bilgi şeridi: video zamanı + son loglanan plakalar
    bar_h = 34
    cv2.rectangle(view, (0, view.shape[0] - bar_h),
                  (view.shape[1], view.shape[0]), (0, 0, 0), -1)
    info = f"{video_sec:6.1f}s  ({realtime_ratio:.1f}x)   Loglanan: "
    info += "  |  ".join(recent_plates[-4:]) if recent_plates else "-"
    cv2.putText(view, info, (10, view.shape[0] - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def process_video(video_path, max_seconds=None, show=True):
    if not os.path.exists(video_path):
        print(f"[HATA] Video bulunamadı: {video_path}")
        return

    model = initialize_model()
    if model is None:
        return
    reader, gpu = initialize_ocr()
    device = "0" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgsz = pick_imgsz(frame_w)
    print(f"\n[VİDEO] {video_path}: {total} kare, {fps:.1f} fps, "
          f"{frame_w}px genişlik → YOLO imgsz={imgsz}, "
          f"her {FRAME_SKIP} karede bir tespit (~{fps/FRAME_SKIP:.0f} Hz)")

    output_dir = "cikti_video"
    os.makedirs(output_dir, exist_ok=True)

    csv_file = open(CSV_PATH, mode='w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Plaka", "Tarih_Saat", "Guven_%", "Video_sn"])

    active_tracks = []
    logged = {}          # {plaka: son loglanan video saniyesi}
    recent_plates = []   # izleme penceresi alt şeridi için
    plate_count = 0
    sample_idx = 0
    frame_no = -1
    t0 = time.time()

    window = None
    if show:
        try:
            window = "Plaka Tespiti (Q/ESC: cik)"
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            print("[UYARI] Görüntü penceresi açılamadı (GUI'siz OpenCV?) "
                  "→ sessiz modda devam")
            window = None

    try:
        while True:
            ok = cap.grab()  # grab: decode etmeden kare atla (hızlı)
            if not ok:
                break
            frame_no += 1
            if max_seconds is not None and frame_no / fps > max_seconds:
                break
            if frame_no % FRAME_SKIP != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            sample_idx += 1
            video_sec = frame_no / fps

            # ─── TESPİT + İZ EŞLEŞTİRME ───
            detections = detect_boxes_fast(model, frame, device, imgsz)
            matched_tracks = set()
            for det in detections:
                box = det[:4]
                best_track, best_iou = None, IOU_MATCH
                for tr in active_tracks:
                    if tr.id in matched_tracks:
                        continue
                    iou = _iou(box, tr.box)
                    if iou > best_iou:
                        best_track, best_iou = tr, iou
                if best_track is None:
                    best_track = PlateTrack(box, sample_idx, video_sec)
                    active_tracks.append(best_track)
                else:
                    best_track.update(box, sample_idx, video_sec)
                matched_tracks.add(best_track.id)
                best_track.add_crop(frame, box, det[4], video_sec)

                # ─── ARTIMLI OKUMA (KONSENSÜS ERKEN ÇIKIŞI) ───
                # İz henüz açıkken, yeterli yeni kırpım biriktikçe OCR
                # tekrar denenir. Tam önümüzde uzun süre giden bir araç
                # hiç kaybolmadığı için izi hiç KAPANMAYABİLİR — eskiden
                # bu durumda sonuç asla üretilmiyordu (ekranda kutu var
                # ama okuma yok). En az 2 yeni kırpım birikmeden tekrar
                # denenmez (CROP_GAP≈0.4sn spacing ile doğal ~0.8sn+
                # throttle sağlar, her örnekte pahalı OCR çalışmaz).
                if not best_track.resolved:
                    n = len(best_track.crops)
                    first_try = best_track.last_attempt_count == 0 and n >= 3
                    if first_try or n - best_track.last_attempt_count >= 2:
                        best_track.last_attempt_count = n
                        result = attempt_read(best_track, reader, output_dir,
                                              "Canlı kontrol")
                        if result and result['cross_validated']:
                            # Çapraz doğrulama sağlanır sağlanmaz iz
                            # "çözülmüş" sayılır — CSV'ye yazılsın ya da
                            # tekrar (dedup) olarak elensin fark etmez;
                            # cevap zaten bilindiğinden artık bu iz için
                            # tekrar tekrar pahalı OCR çalıştırmaya gerek
                            # yok (gerçek hata: aynı araç, dedup'a düşen
                            # bir plaka olduğu için resolved hiç
                            # işaretlenmeyip 50 kırpıma kadar sürekli
                            # yeniden OCR'landı — israf).
                            best_track.resolved = True
                            best_track.resolved_text = result['text']
                            written = log_result(best_track, result, logged,
                                                 csv_writer, csv_file, output_dir)
                            if written:
                                plate_count += 1
                                recent_plates.append(written)

            # ─── SÜRESİ DOLAN İZLERİ KAPAT ───
            # Artık yalnızca HENÜZ ÇÖZÜLMEMİŞ izler için son bir deneme
            # yapılır (canlı kontrol sırasında zaten çapraz doğrulanıp
            # loglanmış izler burada tekrar işlenmez).
            still_active = []
            for tr in active_tracks:
                if sample_idx - tr.last_seen > MISS_LIMIT:
                    if not tr.resolved:
                        result = attempt_read(tr, reader, output_dir, "İz kapandı")
                        written = log_result(tr, result, logged, csv_writer,
                                             csv_file, output_dir) if result else None
                        if written:
                            plate_count += 1
                            recent_plates.append(written)
                else:
                    still_active.append(tr)
            active_tracks = still_active

            # ─── CANLI İZLEME PENCERESİ ───
            if window is not None:
                elapsed = max(0.1, time.time() - t0)
                view = _draw_overlay(frame, active_tracks, recent_plates,
                                     video_sec, video_sec / elapsed)
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("\n[İZLEME] Kullanıcı çıkışı — açık izler "
                          "kapatılıp kayıt sonlandırılıyor...")
                    break

            # ─── İLERLEME ───
            if sample_idx % 300 == 0:
                pct = 100.0 * frame_no / max(1, total)
                rate = frame_no / max(0.1, time.time() - t0)
                eta = (total - frame_no) / max(1.0, rate)
                print(f"[İLERLEME] %{pct:.1f} (video {video_sec:.0f}s) | "
                      f"aktif iz: {len(active_tracks)} | "
                      f"loglanan plaka: {plate_count} | "
                      f"kalan ~{eta/60:.1f} dk")

        # Video bitti: açık kalan, henüz çözülmemiş izler için son deneme
        for tr in active_tracks:
            if not tr.resolved:
                result = attempt_read(tr, reader, output_dir, "Video bitti")
                if result and log_result(tr, result, logged, csv_writer,
                                         csv_file, output_dir):
                    plate_count += 1
    finally:
        cap.release()
        csv_file.close()
        if window is not None:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[BİTTİ] {frame_no + 1} kare ({sample_idx} örnek) "
          f"{elapsed/60:.1f} dakikada işlendi")
    print(f"[BİTTİ] {plate_count} plaka '{CSV_PATH}' dosyasına yazıldı; "
          f"kanıt kırpımları: {output_dir}/")


if __name__ == "__main__":
    video = "test_video2_4k.webm"
    limit = None
    show = True
    for arg in sys.argv[1:]:
        if arg.lower() in ("sessiz", "--sessiz", "headless"):
            show = False
        elif arg.replace(".", "", 1).isdigit():
            limit = float(arg)
        else:
            video = arg
    process_video(video, max_seconds=limit, show=show)
