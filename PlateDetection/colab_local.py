"""
=============================================================================
TÜRK PLAKA TESPİT & TEMİZLEME PİPELINE'I
=============================================================================
Bu modül, araç fotoğraflarından plaka tespiti yapar ve plaka üzerindeki
kirletici unsurları (turuncu muayene sticker'ı, mavi TR bandı, galeri altlığı,
siyah sticker vb.) akıllıca temizleyerek en saf haliyle OCR'a gönderir.

Pipeline Adımları:
  1. YOLO ile plaka bounding box tespiti
  2. Mavi TR bandı tespiti ve kırpma
  3. Turuncu muayene sticker'ı tespiti ve koşullu temizleme
  4. Siyah sticker/yapıştırma tespiti ve temizleme
  5. Plaka altlığı (galeri/bayi yazısı) ve çerçeve temizleme
  6. OCR için görüntü iyileştirme (büyütme, kontrast, keskinleştirme)
  7. Adaptif binarizasyon (farklı aydınlatma koşullarına uyum)
  8. EasyOCR ile metin okuma
  9. Türk plaka formatına düzeltme
  10. CSV loglama ve toplu işleme
"""

import cv2
import numpy as np
import re
import glob
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import easyocr
import torch


# =============================================================================
# BÖLÜM 0: MODEL VE OCR MOTORU BAŞLATMA
# =============================================================================

def initialize_model():
    """
    YOLO modelini yükler. Önce mevcut dizinde 'PlateDetection.pt' arar,
    bulamazsa runs/ altındaki en son eğitim çıktısını (ham, henüz
    adlandırılmamış "best.pt") kullanır.

    Returns:
        model: YOLO model nesnesi veya None (bulunamazsa)
    """
    if os.path.exists("PlateDetection.pt"):
        best_weight = "PlateDetection.pt"
    else:
        # Eğitim çıktılarını tara — en son tarihli olanı al
        weight_files = glob.glob("runs/detect/train*/weights/best.pt")
        best_weight = sorted(weight_files)[-1] if weight_files else None

    if best_weight:
        print(f"[MODEL] Ağırlık dosyası yüklendi: {best_weight}")
        return YOLO(best_weight)
    else:
        print("[HATA] 'PlateDetection.pt' bulunamadı. Eğitim tamamlanmamış olabilir.")
        return None


def initialize_ocr():
    """
    EasyOCR okuyucusunu başlatır.
        
    'en' (İngilizce) karakter seti kullanıyoruz çünkü:
    - Türk plakaları Latin harfleri kullanır
    - EasyOCR'da 'tr' dil desteği plaka için gereksiz ek yük oluşturur
    - allowlist ile zaten sadece plakada olabilecek karakterlere kısıtlıyoruz
    
    Returns:
        reader: EasyOCR Reader nesnesi
        gpu_available: GPU kullanılıp kullanılmadığı (bool)
    """
    gpu_available = torch.cuda.is_available()
    print(f"[OCR] EasyOCR başlatılıyor (GPU: {'Evet' if gpu_available else 'Hayır'})...")
    reader = easyocr.Reader(['en'], gpu=gpu_available)
    return reader, gpu_available


# =============================================================================
# BÖLÜM 1: MAVİ TR BANDI TESPİTİ VE KIRPMA
# =============================================================================

def detect_blue_band(plate_crop):
    """
    Türk plakalarındaki sol taraftaki mavi TR/AB bandını tespit eder ve BEYAZLA BOYAR.
    
    ┌──────────────────────────┐
    │ TR │  34  ABC  1234      │   ← Mavi bant sol tarafta
    └──────────────────────────┘
    
    ÖNCEKİ YAKLAŞIM (SORUNLU): Mavi bandı KIRPIYORDUK
    - Sorun: Mavi bandın hemen sağındaki il kodu rakamları ("34", "42")
      bandın kenarına çok yakın olabilir
    - Kırpma bu rakamları da kesiyordu → "42 AEH 738" → "AEH 738" oluyordu
    - İl kodu kaybı, plaka okumanın en kritik hatası
    
    YENİ YAKLAŞIM: Mavi bandı BEYAZLA BOYAMA (Inpaint)
    - Mavi pikselleri tespit et → beyaza boya
    - İl kodu rakamları yerinde kalır, sadece mavi arka plan temizlenir
    - Siyah karakterler mavi bant üzerinde bile okunabilir hale gelir
    
    NASIL ÇALIŞIR:
    1. Görüntüyü BGR → HSV renk uzayına çeviririz.
       - HSV (Hue-Saturation-Value) renk tespiti için en uygun uzaydır
       - BGR'de aydınlatma değişimi renk algılamayı çok bozar
       - HSV'de Hue kanalı aydınlatmadan bağımsızdır
    
    2. Sadece plakanın sol %25'ine bakarız (mavi bant her zaman soldadır).
       - Tüm plakada mavi aramak false positive verir
       - Mavi renkli araba gövdesi, gökyüzü yansıması vb. karışır
    
    3. HSV'de mavi renk aralığını tanımlarız:
       - Hue: 100-130 (mavi tonlar, OpenCV'de 0-180 aralığında)
       - Saturation: 80+ (canlı mavi, soluk/gri tonlar hariç)
       - Value: 50+ (çok karanlık pikselleri hariç tut)
    
    4. Mavi piksel oranı eşik değerini aşarsa bant var deriz.
       - Eşik: sol bölgenin %8'i
    
    5. Bant varsa, mavi pikselleri beyaza boyarız (inpaint).
       - Karakterler (siyah) korunur, sadece mavi arka plan temizlenir
    
    Parametreler:
        plate_crop (np.ndarray): BGR formatında kırpılmış plaka görüntüsü
    
    Returns:
        result_image (np.ndarray): Mavi bant temizlenmiş (veya orijinal) görüntü
        band_detected (bool): Mavi bant tespit edildi mi?
        band_width (int): Tespit edilen bant genişliği (piksel)
    """
    h, w = plate_crop.shape[:2]
    result = plate_crop.copy()
    
    # Sadece sol %25'e bak — mavi bant her zaman burada
    left_w = int(w * 0.25)
    left_region = plate_crop[:, :left_w]
    
    # BGR → HSV dönüşümü
    # HSV renk uzayı: Hue (renk tonu), Saturation (doygunluk), Value (parlaklık)
    hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
    
    # Mavi renk aralığı (HSV)
    # Hue: 100-130 → Mavi tonlar (OpenCV'de Hue 0-180 aralığında, normal 360°'nin yarısı)
    # Saturation: 80-255 → Canlı mavi (düşük saturasyon = gri/soluk = istemediğimiz)
    # Value: 50-255 → Orta-parlak (çok karanlık pikseller gürültü olabilir)
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([130, 255, 255])
    
    # inRange: her pikseli kontrol eder, aralık içindeyse 255 (beyaz), değilse 0 (siyah)
    # Sonuç: mavi piksellerin bulunduğu bir maske (binary görüntü)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Mavi piksel oranını hesapla
    # countNonZero: maskede beyaz (255) olan piksel sayısını verir
    blue_pixel_count = cv2.countNonZero(blue_mask)
    total_pixels = left_region.shape[0] * left_region.shape[1]
    blue_ratio = blue_pixel_count / total_pixels if total_pixels > 0 else 0
    
    # Eşik değeri: sol bölgenin %8'inden fazlası maviyse bant var
    BLUE_BAND_THRESHOLD = 0.08
    
    if blue_ratio > BLUE_BAND_THRESHOLD:
        # Mavi bölgenin en sağ sınırını bul (istatistik için)
        cols_with_blue = np.where(blue_mask > 0)
        band_width = 0
        if len(cols_with_blue[1]) > 0:
            band_width = int(np.max(cols_with_blue[1])) + 1
        
        # ─── YENİ: KIRPMA YERİNE BOYAMA ───
        # Tam boyutlu maske oluştur (tüm plaka boyutunda)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        # Sol bölgedeki mavi maskeyi tam boyutlu maskeye kopyala
        full_mask[:, :left_w] = blue_mask
        
        # Maskeyi biraz genişlet — mavi-beyaz geçiş pikselleri için
        full_mask = cv2.dilate(full_mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Inpaint: mavi bölgeyi çevresindeki renklerle doldur
        # Bu, karakterleri (siyah) korur ve mavi arka planı beyaza çevirir
        result = cv2.inpaint(result, full_mask, 5, cv2.INPAINT_TELEA)
        
        print(f"    [MAVİ BANT] Tespit edildi! Genişlik: {band_width}px, "
              f"Oran: {blue_ratio:.1%} → Beyaza boyandı (kırpma yok)")
        return result, True, band_width
    
    print(f"    [MAVİ BANT] Tespit edilmedi (mavi oran: {blue_ratio:.1%})")
    return result, False, 0


# =============================================================================
# BÖLÜM 2: TURUNCU MUAYENE STİCKER'I TESPİTİ VE KOŞULLU TEMİZLEME
# =============================================================================

def detect_and_remove_orange_sticker(plate_crop):
    """
    Türk plakalarındaki turuncu araç muayene sticker'ını tespit eder.
    SADECE gerçekten tespit edilirse temizler — yoksa dokunmaz.
    
    ┌──────────────────────────┐
    │  34 🟠 ABC  1234         │   ← Turuncu sticker (muayene damgası)
    └──────────────────────────┘
    
    ESKİ KODDAKİ SORUN:
    Eski kod HER plakaya turuncu temizleme uyguluyordu. Turuncu sticker
    olmayan plakalarda inpaint algoritması beyaz plaka yüzeyinde gereksiz
    artefaktlar oluşturuyordu (bulanık lekeler, renk kayması).
    
    YENİ YAKLAŞIM:
    1. Önce turuncu piksel sayısını ölç
    2. Eşik altındaysa "sticker yok" de ve orijinali döndür
    3. Eşik üstündeyse inpaint ile temizle
    
    NASIL ÇALIŞIR:
    1. BGR → HSV dönüşümü (renk tespiti için)
    
    2. Turuncu renk aralığı tanımlama:
       - Hue: 5-25 → Turuncu tonlar
         (0-10 arası kırmızıya yakın, 20-30 arası sarıya yakın)
       - Saturation: 100-255 → Canlı turuncu
         (düşük saturasyon = bej/krem rengi = plaka yüzeyi, istemiyoruz)
       - Value: 100-255 → Parlak turuncu
         (düşük value = koyu kahverengi tonu = istemiyoruz)
    
    3. Piksel sayma ve eşik kontrolü:
       - Toplam plaka pikselinin %1'inden fazlası turuncuysa sticker var
       - %1 eşiği küçük gibi görünebilir ama turuncu sticker plaka alanının
         yaklaşık %2-5'ini kaplar — %1 güvenli bir alt sınır
    
    4. Morfolojik genişletme (dilate):
       - Tespit edilen turuncu maskeyi biraz büyütürüz
       - Neden? Sticker'ın kenarları yarı-turuncu olabilir (HSV eşiği kaçırır)
       - 5x5 kernel, 2 iterasyon → ~10 piksel genişleme
       - Bu, inpaint'in sticker kenarlarını da temizlemesini sağlar
    
    5. Inpainting (cv2.inpaint):
       - TELEA algoritması: Fast Marching Method tabanlı
       - Maskelenen bölgeyi çevresindeki piksellere bakarak doldurur
       - Yarıçap 3: her maskelenen piksel, 3 piksel mesafedeki komşulara bakar
       - Küçük yarıçap = daha keskin sonuç (plaka yüzeyi düz olduğu için yeterli)
    
    Parametreler:
        plate_crop (np.ndarray): BGR formatında plaka görüntüsü
    
    Returns:
        result_image (np.ndarray): Temizlenmiş veya orijinal görüntü
        sticker_found (bool): Turuncu sticker tespit edildi mi?
    """
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    
    # Turuncu renk aralığı (HSV)
    # Not: OpenCV'de Hue 0-180 aralığında (360°'nin yarısı)
    # Turuncu renk: gerçek dünyada ~15-45° → OpenCV'de ~7-22
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Turuncu piksellerin maskesini oluştur
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Turuncu piksel oranını hesapla
    orange_pixels = cv2.countNonZero(orange_mask)
    total_pixels = plate_crop.shape[0] * plate_crop.shape[1]
    orange_ratio = orange_pixels / total_pixels if total_pixels > 0 else 0
    
    # ─── ANA KARAR: Sticker var mı? ───
    # Eşik: toplam plaka alanının %0.5'i
    # Neden %0.5?
    # - Turuncu sticker genelde plaka alanının %2-5'ini kaplar
    # - Mavi bant boyama sonrası sticker'ın oranı düşebilir (piksel kaybı)
    # - %0.5 = düşük eşik, küçük veya kısmen görünen sticker'ları da yakalar
    # - %0.5'den az turuncu piksel = muhtemelen gürültü veya araba gövdesinden yansıma
    # - Test sonucu: araba1.jpg'de sticker %0.82 oranındaydı, %1 eşiği kaçırıyordu
    ORANGE_THRESHOLD = 0.005
    
    if orange_ratio < ORANGE_THRESHOLD:
        print(f"    [TURUNCU] Sticker tespit EDİLMEDİ (oran: {orange_ratio:.3%}) → Dokunulmuyor")
        return plate_crop.copy(), False
    
    print(f"    [TURUNCU] Sticker tespit EDİLDİ! (oran: {orange_ratio:.3%}) → Temizleniyor")
    
    # Morfolojik genişletme (dilation)
    # ─────────────────────────────────
    # np.ones((5,5)): 5x5 piksellik kare yapısal eleman (kernel)
    # iterations=2: genişletmeyi 2 kez tekrarla
    # 
    # Neden genişletme yapıyoruz?
    # HSV eşikleme sticker'ın tam sınırını yakalayamayabilir çünkü:
    # - Sticker kenarları yarı saydam olabilir
    # - JPEG sıkıştırma kenar piksellerde renk karışımı oluşturur
    # - Aydınlatma kenar bölgelerde renk kaymasına neden olur
    # Genişletme, maskeyi birkaç piksel büyüterek bu kenar problemlerini çözer
    kernel = np.ones((5, 5), np.uint8)
    orange_mask_dilated = cv2.dilate(orange_mask, kernel, iterations=2)
    
    # Inpainting ile turuncu bölgeyi doldur
    # ─────────────────────────────────────
    # cv2.inpaint(kaynak, maske, yarıçap, yöntem)
    # - kaynak: orijinal BGR görüntü
    # - maske: doldurulacak bölge (beyaz = doldur, siyah = dokunma)
    # - yarıçap (3): her doldurulan piksel için komşuluk mesafesi
    #   → 3 piksel = küçük yarıçap = keskin sonuç (plaka düz yüzey)
    #   → Büyük yarıçap daha bulanık ama daha pürüzsüz sonuç verir
    # - INPAINT_TELEA: Alexandru Telea'nın Fast Marching Method algoritması
    #   → Maskenin kenarından içeriye doğru yayılarak doldurur
    #   → Alternatif: INPAINT_NS (Navier-Stokes) — daha yavaş ama dokular için daha iyi
    #   → Plaka düz beyaz yüzey olduğu için TELEA yeterli ve daha hızlı
    result = cv2.inpaint(plate_crop, orange_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    return result, True


# =============================================================================
# BÖLÜM 3: SİYAH STİCKER / YAPIŞTIRMA TESPİTİ VE TEMİZLEME
# =============================================================================

def detect_and_remove_black_sticker(plate_crop):
    """
    Plaka üzerindeki siyah yapıştırmaları (vignette sticker, vergi damgası vb.)
    tespit eder ve temizler. Plaka karakterlerinden ayırt eder.
    
    KRİTİK ZORLUK:
    Plaka karakterleri de siyahtır! Siyah sticker'ı siyah harflerden ayırmamız gerekir.
    
    AYIRT ETME STRATEJİSİ:
    - Plaka karakterleri: düşey dikdörtgensel, belirli aspect ratio (0.3-0.9)
    - Siyah sticker: genelde yuvarlak, kare veya düzensiz şekil
    - Sticker genellikle plakanın kenar bölgelerinde bulunur (köşeler)
    - Sticker'ların alanı karakterlerden farklıdır (çok büyük veya çok küçük)
    
    NASIL ÇALIŞIR:
    1. Gri tonlamaya çevir ve OTSU eşikleme ile binarize et
    2. Tüm konturları bul
    3. Her kontur için:
       a. Aspect ratio hesapla (en/boy oranı)
       b. Bounding rect alanı hesapla
       c. Konum kontrolü (plaka kenarlarında mı?)
       d. Dairesellik kontrolü (circularity)
    4. Karakter olmayan siyah bölgeleri maskelene ekle
    5. Inpaint ile temizle
    
    Parametreler:
        plate_crop (np.ndarray): BGR formatında plaka görüntüsü
    
    Returns:
        result_image (np.ndarray): Temizlenmiş veya orijinal görüntü
        sticker_found (bool): Siyah sticker tespit edildi mi?
    """
    h, w = plate_crop.shape[:2]
    plate_area = h * w
    
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # OTSU eşikleme
    # ──────────────
    # OTSU: Otomatik eşik değeri bulma algoritması
    # - Görüntüdeki piksel dağılımını (histogram) analiz eder
    # - Sınıflar-arası varyansı (between-class variance) maksimize eden eşiği bulur
    # - Yani: arka plan ve ön plan piksellerini en iyi ayıran değeri otomatik seçer
    # - Sabit eşik (örn. 127) farklı aydınlatma koşullarında başarısız olur
    # - THRESH_BINARY_INV: eşik altı → beyaz, üstü → siyah (siyah nesneleri yakalar)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Konturları bul
    # ──────────────
    # RETR_EXTERNAL: sadece en dış konturları al (iç içe konturları atla)
    #   → Harflerin iç boşlukları (O, A, D harflerinin ortası) ayrı kontur olmasın
    # CHAIN_APPROX_SIMPLE: kontur noktalarını sadeleştir (hafıza tasarrufu)
    #   → Düz çizgiler üzerindeki ara noktaları atar, sadece köşeleri tutar
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sticker maskesi — temizlenecek bölgeleri biriktireceğiz
    sticker_mask = np.zeros((h, w), dtype=np.uint8)
    sticker_count = 0
    
    # Karakter boyut referansları (Türk plaka oranlarına göre)
    # Türk plakası: 520mm x 110mm → aspect ratio ~4.7:1
    # Tek karakter yaklaşık: genişlik = plaka_w / 10, yükseklik = plaka_h * 0.6
    expected_char_h_min = h * 0.25  # Karakterin minimum yüksekliği
    expected_char_h_max = h * 0.85  # Karakterin maximum yüksekliği
    expected_char_w_min = w * 0.02  # Karakterin minimum genişliği
    expected_char_w_max = w * 0.15  # Karakterin maximum genişliği
    
    for contour in contours:
        # Bounding rectangle: konturu çevreleyen en küçük dik dikdörtgen
        x, y, cw, ch = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        rect_area = cw * ch
        
        if rect_area == 0 or contour_area == 0:
            continue
        
        # ─── FİLTRELEME KRİTERLERİ ───
        
        # 1. Aspect Ratio (en/boy oranı)
        # Plaka karakterleri düşey dikdörtgenseldir: aspect ratio ~0.4-0.8
        # Yuvarlak sticker: aspect ratio ~0.8-1.2
        # Yatay uzun sticker: aspect ratio > 1.5
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # 2. Dairesellik (Circularity)
        # Formül: 4π × alan / çevre²
        # Mükemmel daire = 1.0, kare ≈ 0.78, düzensiz şekil < 0.5
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 3. Konum kontrolü — sticker genellikle köşelerde veya kenarlarda
        is_edge = (x < w * 0.1 or x + cw > w * 0.9 or  # Sol/sağ kenar
                   y < h * 0.1 or y + ch > h * 0.9)       # Üst/alt kenar
        
        # 4. Boyut kontrolü — karakter boyut aralığında mı?
        is_char_sized = (expected_char_h_min <= ch <= expected_char_h_max and
                         expected_char_w_min <= cw <= expected_char_w_max)
        
        # 5. Alan oranı — plaka alanına göre çok küçük veya çok büyük kontroller
        area_ratio = rect_area / plate_area
        
        # ─── KARAR: Bu kontur sticker mı? ───
        # Sticker olma koşulları (en az biri doğruysa):
        is_sticker = False
        
        # Koşul A: Yüksek dairesellik + kenarda (yuvarlak sticker)
        if circularity > 0.65 and is_edge and area_ratio > 0.005:
            is_sticker = True
        
        # Koşul B: Karakter boyutu dışında + kenarda
        if not is_char_sized and is_edge and area_ratio > 0.005 and area_ratio < 0.25:
            # Çok yuvarlak veya çok kare (karakter aspect ratio'su dışında)
            if aspect_ratio > 0.85 or aspect_ratio < 0.2:
                is_sticker = True
        
        # Koşul C: Plakanın alt köşelerindeki küçük yapışmalar
        if y > h * 0.7 and area_ratio < 0.08 and area_ratio > 0.002:
            if not (0.3 <= aspect_ratio <= 0.9 and expected_char_h_min <= ch):
                is_sticker = True
        
        if is_sticker:
            cv2.drawContours(sticker_mask, [contour], -1, 255, -1)
            sticker_count += 1
    
    if sticker_count > 0:
        print(f"    [SİYAH] {sticker_count} adet siyah sticker/yapıştırma tespit edildi → Temizleniyor")
        # Maskeyi biraz genişlet (kenar pikselleri temizlemek için)
        sticker_mask = cv2.dilate(sticker_mask, np.ones((3, 3), np.uint8), iterations=1)
        result = cv2.inpaint(plate_crop, sticker_mask, 3, cv2.INPAINT_TELEA)
        return result, True
    else:
        print(f"    [SİYAH] Siyah sticker tespit edilmedi")
        return plate_crop.copy(), False


# =============================================================================
# BÖLÜM 4: PLAKA ALTLIĞI VE ÇERÇEVE TEMİZLEME
# =============================================================================

def remove_plate_frame_and_holder(plate_crop):
    """
    Plaka altlığı (galeri/bayi yazısı) ve çerçeveyi temizler.
    
    Türk plakalarında sık görülen kirletici unsurlar:
    ┌──────────────────────────────┐
    │         34 EVC 531           │  ← Plaka metni
    │      GALERİ OSMAN            │  ← Altlık yazısı (temizlenecek)
    └──────────────────────────────┘
    ↑                              ↑
    Çerçeve kenarları (temizlenecek)
    
    NASIL ÇALIŞIR:
    
    Adım 1 — Plaka Sınırını Bul (Kontur Analizi):
    - Gri tonlama + GaussianBlur + Canny kenar tespiti
    - En büyük dikdörtgensel konturu bul = plaka sınırı
    - Bu dikdörtgenin içini kırp → çerçeve dışarıda kalır
    
    Adım 2 — Alt Bölge Temizleme (Altlık Yazısı):
    - Plakanın alt %25'ini incele
    - Bu bölgedeki küçük yazıları (galeri adı) beyazla kapat
    - Ana plaka karakterlerinden küçük olan her şeyi temizle
    
    Parametreler:
        plate_crop (np.ndarray): BGR formatında plaka görüntüsü
    
    Returns:
        result_image (np.ndarray): Temizlenmiş görüntü
    """
    h, w = plate_crop.shape[:2]
    result = plate_crop.copy()
    
    # ─── ADIM 1: ÇERÇEVE TEMİZLEME (Kontur tabanlı kırpma) ───
    
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # GaussianBlur: gürültüyü azalt
    # ────────────────────────────────
    # (5,5): kernel boyutu — 5x5 piksellik alan üzerinde ortalama alır
    # 0: sigma değeri — otomatik hesaplanır (kernel boyutundan türetilir)
    # Neden blur? Canny kenar tespiti gürültüye çok hassastır.
    # Blur olmadan, JPEG artefaktları ve piksel gürültüsü sahte kenarlar oluşturur.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Kenar Tespiti
    # ───────────────────
    # İki eşik değeri kullanır (hysteresis thresholding):
    # - 50 (düşük eşik): bu değerin altındaki gradyanlar kesinlikle kenar değil
    # - 200 (yüksek eşik): bu değerin üstündeki gradyanlar kesinlikle kenar
    # - Aradaki gradyanlar: sadece yüksek eşikli bir kenara bağlıysa kenar sayılır
    # Bu iki eşikli yaklaşım, tek eşikli yöntemlerden çok daha az gürültü yakalar
    edges = cv2.Canny(blurred, 50, 200)
    
    # Konturları bul ve en büyük dikdörtgensel olanı seç
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    best_rect_area = 0
    
    for contour in contours:
        # approxPolyDP: konturu daha az noktalı bir poligona yaklaştırır
        # ──────────────────────────────────────────────────────────────
        # epsilon: yaklaşım hassasiyeti = çevre × 0.02
        # - Küçük epsilon = orijinale yakın (çok noktalı)
        # - Büyük epsilon = basitleştirilmiş (az noktalı)
        # - 0.02: dikdörtgensel şekilleri 4 noktalı poligona indirger
        # True: kontur kapalı (başlangıç = bitiş noktası)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 4 noktalı poligon = dikdörtgen (plaka sınırı adayı)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Minimum alan kontrolü: plaka alanının en az %30'u olmalı
            # (çok küçük dikdörtgenler = vida deliği, kenar parçası vb.)
            if area > plate_area_min(h, w) and area > best_rect_area:
                best_rect = approx
                best_rect_area = area
    
    if best_rect is not None:
        # Bulunan dikdörtgenin bounding rect'ini al ve kırp
        rx, ry, rw, rh = cv2.boundingRect(best_rect)
        # Güvenlik kontrolü: kırpılan alan orijinalin en az %50'si olmalı
        if rw > w * 0.5 and rh > h * 0.5:
            result = result[ry:ry+rh, rx:rx+rw]
            h, w = result.shape[:2]  # Boyutları güncelle
    
    # ─── ADIM 2: ALT BÖLGE TEMİZLEME (Galeri/bayi yazısı) ───
    
    # Plakanın alt %25'ini incele — altlık yazıları burada bulunur
    # Neden %25? Galeri yazıları genellikle plakanın alt 1/4'ünde yer alır.
    # Daha büyük oran ana karakterlere müdahale riski taşır.
    bottom_start = int(h * 0.75)
    bottom_region = result[bottom_start:, :]
    
    if bottom_region.shape[0] > 5 and bottom_region.shape[1] > 5:
        bottom_gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Thresholding: yerel bölge bazlı eşikleme
        # ──────────────────────────────────────────────────
        # ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian ağırlıklı yerel ortalama
        # THRESH_BINARY_INV: eşik altı → beyaz (koyu pikselleri yakala)
        # 11: blok boyutu (11x11 piksellik komşuluk bölgesi)
        # 2: sabit çıkarma değeri (hassasiyet ayarı)
        bottom_binary = cv2.adaptiveThreshold(
            bottom_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Alt bölgedeki konturları bul
        bottom_contours, _ = cv2.findContours(
            bottom_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Ana plaka karakterlerinin beklenen minimum yüksekliği
        # Alt bölgedeki yazılar ana karakterlerden çok daha küçüktür
        # Ana karakter yüksekliği ≈ plaka yüksekliğinin %40-70'i
        # Altlık yazısı ≈ plaka yüksekliğinin %5-20'si
        main_char_min_h = h * 0.30
        
        for contour in bottom_contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Ana karakter yüksekliğinden KÜÇÜK olan her şeyi beyazla kapat
            # (galeri adı, telefon numarası vb.)
            if ch < main_char_min_h:
                # Alt bölgedeki koordinatları tam görüntü koordinatlarına çevir
                abs_y = bottom_start + y
                # Beyaz dikdörtgen çiz (plaka arka plan rengine uygun)
                cv2.rectangle(result, (x, abs_y), (x + cw, abs_y + ch), (255, 255, 255), -1)
    
    return result


def plate_area_min(h, w):
    """Minimum kabul edilebilir plaka alan hesabı (toplam alanın %30'u)"""
    return h * w * 0.30


# =============================================================================
# BÖLÜM 5: OCR İÇİN GÖRÜNTÜ İYİLEŞTİRME
# =============================================================================

def enhance_plate_for_ocr(plate_crop):
    """
    Plaka görüntüsünü OCR için optimize eder.
    Düşük çözünürlüklü, bulanık veya düşük kontrastlı plakaları okunabilir hale getirir.
    
    İŞLEM SIRASI VE NEDENLERİ:
    
    1. BÜYÜTME (4x) — Neden?
       OCR motorları minimum ~32px karakter yüksekliği ister.
       Bir plaka kırpması 30-80px yüksekliğinde olabilir.
       4x büyütme ile 120-320px'e çıkar → OCR doğruluğu dramatik artar.
    
    2. CLAHE (Kontrast Normalizasyonu) — Neden?
       Gölgede veya ters ışıkta çekilen plakalarda kontrast çok düşük olabilir.
       Normal histogram eşitleme tüm görüntüyü aynı anda işler → aşırı kontrast.
       CLAHE yerel bölgelerde (tile) çalışır → dengeli kontrast artışı.
    
    3. BİLATERAL FİLTRE (Gürültü Azaltma) — Neden?
       Normal blur (Gaussian) kenarları da bulanıklaştırır → karakterler erir.
       Bilateral filtre kenarları KORUR, sadece düz alanları yumuşatır.
       Bu, karakter kenarlarının keskin kalmasını sağlar.
    
    4. UNSHARP MASKING (Keskinleştirme) — Neden?
       Büyütme ve filtreleme sonrası görüntü biraz yumuşak kalabilir.
       Unsharp masking: bulanık versiyonu orijinalden çıkararak kenarları vurgular.
       Ağırlık 1.5: orijinalin 1.5 katı - bulanığın 0.5 katı = keskinleştirilmiş.
    
    Parametreler:
        plate_crop (np.ndarray): BGR formatında plaka görüntüsü
    
    Returns:
        enhanced (np.ndarray): İyileştirilmiş görüntü
    """
    # ─── 1. BÜYÜTME (ADAPTİF ÖLÇEK) ───  [DEĞİŞTİ]
    # ESKİ: sabit 4x büyütme
    #   - 30px'lik kırpım → 120px (yeterli) ama 150px'lik kırpım → 600px
    #   - Aşırı büyütme JPEG artefaktlarını da büyütür + işlemi yavaşlatır
    # YENİ: hedef yüksekliğe göre ölçek hesapla
    #   - Hedef: ~160px plaka yüksekliği → karakter yüksekliği ~90-110px
    #     (EasyOCR'ın CRAFT dedektörü için ideal aralık)
    #   - Zaten büyük kırpımlar gereksiz yere şişirilmez
    # INTER_LANCZOS4: 8x8 komşuluk kullanan en kaliteli interpolasyon
    #   - INTER_CUBIC'ten (4x4) daha keskin kenar üretir
    #   - Küçük plakaların büyütülmesinde karakter netliği kritik
    TARGET_PLATE_HEIGHT = 160
    h0 = plate_crop.shape[0]
    scale = TARGET_PLATE_HEIGHT / max(h0, 1)
    scale = min(max(scale, 1.0), 8.0)  # 1x-8x aralığına sınırla
    plate_large = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)
    
    # ─── 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) ───
    # Normal histogram eşitleme vs CLAHE:
    # - Normal: tüm görüntüyü tek histogram olarak işler → aşırı kontrast
    # - CLAHE: görüntüyü küçük karelere (tile) böler, her birini ayrı eşitler
    #
    # clipLimit=2.0: kontrast sınırı
    #   → Bir tile'daki histogram kutucuğu bu değeri aşarsa kesilir ve dağıtılır
    #   → Düşük değer = az kontrast artışı (1.0), yüksek = agresif artış (4.0+)
    #   → 2.0 = plakalar için dengeli değer
    #
    # tileGridSize=(8,8): görüntüyü 8x8=64 kareye böl
    #   → Her kare ayrı ayrı histogram eşitlemesi alır
    #   → Küçük grid = daha yerel adaptasyon (ama daha çok gürültü amplifikasyonu)
    #   → Büyük grid = daha global (ama adaptif avantajı azalır)
    
    # CLAHE sadece tek kanallı (gri) görüntüde çalışır
    # LAB renk uzayında L (Lightness) kanalına uygulayalım — renkleri bozmaz
    lab = cv2.cvtColor(plate_large, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    plate_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # ─── 3. BİLATERAL FİLTRE ───
    # cv2.bilateralFilter(kaynak, d, sigmaColor, sigmaSpace)
    #
    # d=9: filtre çapı (piksel)
    #   → Büyük d = daha güçlü yumuşatma ama daha yavaş
    #   → 9 = orta seviye, iyi denge
    #
    # sigmaColor=75: renk uzayı sigma değeri
    #   → Bu değerden büyük renk farkına sahip pikseller karışmaz
    #   → Yüksek değer = daha fazla renk karışır (daha bulanık)
    #   → 75 = karakter kenarlarını (siyah-beyaz geçiş) korur
    #
    # sigmaSpace=75: koordinat uzayı sigma değeri
    #   → Uzaktaki piksellerin etkisini kontrol eder
    #   → Yüksek değer = daha geniş alanda pikseller karışır
    #
    # NEDEN BİLATERAL?
    # Gaussian blur: her piksel eşit ağırlıkla karışır → kenarlar erir
    # Bilateral: sadece benzer renkli pikseller karışır → kenarlar korunur
    # Bu, plaka yüzeyindeki gürültüyü temizlerken karakter kenarlarını korur
    plate_smooth = cv2.bilateralFilter(plate_clahe, 9, 75, 75)
    
    # ─── 4. UNSHARP MASKING (Keskinleştirme) ───
    # Prensip: Keskin = Orijinal + (Orijinal - Bulanık) × α
    # Bulanık versiyonu orijinalden çıkararak kenar detaylarını vurgular
    #
    # GaussianBlur: (0,0) kernel + sigma=3.0 ile bulanık versiyon oluştur
    # (0,0): kernel boyutunu sigma'dan otomatik hesapla
    # addWeighted: ağırlıklı toplam
    #   1.5: orijinalin ağırlığı (1.0 = değişiklik yok, >1 = keskinleştir)
    #   -0.5: bulanığın ağırlığı (negatif = bulanığı çıkar)
    #   0: ek sabit (brightness offset)
    gaussian = cv2.GaussianBlur(plate_smooth, (0, 0), 3.0)
    enhanced = cv2.addWeighted(plate_smooth, 1.5, gaussian, -0.5, 0)

    # ─── 5. KENAR PAYI EKLEME ───  [YENİ]
    # EasyOCR'ın metin dedektörü (CRAFT), görüntü kenarına dayanan
    # karakterleri kaçırabiliyor veya kısmen algılıyor.
    # Görüntünün etrafına 16px'lik kenar payı ekleyerek ilk ve son
    # karakterin de tam algılanmasını garantiliyoruz.
    # BORDER_REPLICATE: kenar piksellerini kopyalar (yapay kontrast
    # çizgisi oluşturmaz — sabit renk dolgu sahte kenar üretebilirdi)
    enhanced = cv2.copyMakeBorder(enhanced, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    return enhanced


# =============================================================================
# BÖLÜM 6: ADAPTİF BİNARİZASYON
# =============================================================================

def adaptive_binarize(plate_image):
    """
    Plaka görüntüsünü siyah-beyaz'a çevirir (binarizasyon).
    Farklı aydınlatma koşullarına otomatik adapte olur.
    
    NEDEN BİNARİZASYON?
    - OCR motorları genellikle yüksek kontrastlı görüntülerde daha iyi çalışır
    - Renkli gürültü (sticker kalıntısı, gölge) tamamen ortadan kalkar
    - İşlem hızı artar (tek kanal vs üç kanal)
    
    SABIT EŞİK vs ADAPTİF EŞİK:
    - Sabit eşik (örn. threshold = 127):
      → Eşit aydınlatılmış görüntülerde iyi
      → Gölge/parıltı olan bölgelerde başarısız (yarısı siyah, yarısı beyaz)
    - Adaptif eşik:
      → Her piksel için yerel komşuluk ortalaması hesaplanır
      → Her bölge kendi aydınlatmasına göre eşiklenir
      → Gölge altındaki bölgeler bile doğru eşiklenir
    
    MORFOLOJİK İŞLEMLER:
    - Açma (Opening = Erosion + Dilation):
      → Küçük beyaz gürültü noktalarını temizler
      → Karakterlerin genel şeklini korur
    - Kapama (Closing = Dilation + Erosion):
      → Karakter içindeki küçük delikleri kapatır
      → Kırık karakterleri birleştirir
    
    Parametreler:
        plate_image (np.ndarray): BGR veya gri tonlama görüntü
    
    Returns:
        binary (np.ndarray): Siyah-beyaz görüntü (tek kanal)
    """
    # Renkli ise gri tonlamaya çevir
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image.copy()
    
    # Adaptive Gaussian Thresholding
    # ──────────────────────────────
    # ADAPTIVE_THRESH_GAUSSIAN_C:
    #   Her piksel için komşuluk bölgesindeki piksellerin Gaussian ağırlıklı
    #   ortalamasını hesaplar ve bunu eşik değeri olarak kullanır.
    #   Gaussian: merkeze yakın pikseller daha çok ağırlık alır
    #
    # THRESH_BINARY:
    #   Piksel > eşik → 255 (beyaz), Piksel <= eşik → 0 (siyah)
    #   (THRESH_BINARY_INV'in tersi — burada karakterler siyah kalır)
    #
    # 255: maximum değer (beyaz)
    # 15: blok boyutu (15x15 piksellik komşuluk)
    #   → Tek sayı olmalı (merkez piksel için simetrik komşuluk)
    #   → Büyük blok = daha geniş alan ortalaması (daha global)
    #   → Küçük blok = daha yerel (daha hassas ama daha gürültülü)
    #   → 15 = plaka karakterleri için dengeli boyut
    # 4: sabit çıkarma değeri (C parametresi)
    #   → Hesaplanan yerel ortalamadan 4 çıkarılır
    #   → Pozitif C = daha sert eşikleme (daha az piksel beyaz olur)
    #   → Negatif C = daha yumuşak eşikleme
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    
    # Morfolojik Açma (Opening)
    # ─────────────────────────
    # Opening = Erosion → Dilation
    # Erosion: her pikseli komşularının MINIMUMU ile değiştirir → küçük beyaz noktalar kaybolur
    # Dilation: kaybedilen boyutu geri kazandırır → ana şekiller korunur
    # Net etki: küçük gürültü noktaları temizlenir, büyük yapılar kalır
    kernel_open = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # Morfolojik Kapama (Closing)
    # ───────────────────────────
    # Closing = Dilation → Erosion
    # Dilation: küçük delikleri ve boşlukları kapatır
    # Erosion: büyüyen boyutu geri çeker
    # Net etki: karakter içindeki küçük delikler kapanır, kırık çizgiler birleşir
    kernel_close = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    return binary


# =============================================================================
# BÖLÜM 7: OCR İLE PLAKA OKUMA
# =============================================================================

def _join_fragments(fragments):
    """Parça listesini soldan sağa birleştir, uzunluk-ağırlıklı güven hesapla."""
    if not fragments:
        return "", 0.0
    fragments = sorted(fragments, key=lambda f: f['x'])
    text = "".join(f['text'] for f in fragments)
    total_len = sum(len(f['text']) for f in fragments)
    # Uzunluk-ağırlıklı güven: uzun parçalar sonucu daha çok belirler
    # (basit ortalama, 1 karakterlik sahte parçaya 6 karakterlik gerçek
    #  parçayla eşit ağırlık veriyordu)
    conf = sum(f['conf'] * len(f['text']) for f in fragments) / total_len
    return text, conf


def merge_ocr_fragments(ocr_result):
    """
    [YENİ FONKSİYON — OCR PARÇALARINI ÇOK HİPOTEZLİ BİRLEŞTİRME]

    EasyOCR plakayı genelde 2-3 parça halinde okur: "06", "ABY", "325".
    Ama araya sahte parçalar da karışır: plaka çerçevesi kenarı, vida,
    TR bandı yazısı, plaka yanındaki galeri sticker'ı...

    Hangi parçanın sahte olduğunu tek bir kuralla bilemeyiz — bu yüzden
    FARKLI FİLTRE KOMBİNASYONLARIYLA BİRDEN FAZLA HİPOTEZ üretilir ve
    hepsi aday olarak skorlamaya gönderilir. Yapısal plaka doğrulaması
    hangisinin doğru olduğuna karar verir.

    HİPOTEZLER:
    1. "yükseklik filtreli": ana metin yüksekliğinin %45'inden kısa
       parçalar atılır (çerçeve kenarı '1' artefaktını çözer — plaka
       karakterleri hep aynı yüksekliktedir)
    2. "yükseklik + güven filtreli": ek olarak güveni 0.30'un altındaki
       parçalar atılır. GERÇEK HATA ÖRNEĞİ: gevşek kırpımda galeri
       sticker'ından 'EN' parçası (güven 0.16) plakaya karışıp
       '42ALDEN013' üretiyordu; gerçek parçalar ('42'=0.74, 'ALD'=1.00,
       '013'=0.65) güven filtresiyle ayrışır → '42ALD013' GEÇERLİ!
    3. "filtresiz": tüm parçalar — yükseklik filtresi yanlışlıkla
       gerçek parça attıysa telafi eder

    Parametreler:
        ocr_result: EasyOCR readtext çıktısı [(bbox, text, conf), ...]

    Returns:
        hypotheses (list): [(text, conf, hipotez_adı), ...] — tekrarsız
    """
    if not ocr_result:
        return []

    fragments = []
    for bbox, text, conf in ocr_result:
        text = text.replace(" ", "")
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        fragments.append({
            'x': min(xs),                  # Soldan sıralama için
            'h': max(ys) - min(ys),        # Parça yüksekliği (filtre için)
            'text': text,
            'conf': conf,
        })

    if not fragments:
        return []

    max_h = max(f['h'] for f in fragments)
    tall = [f for f in fragments if f['h'] >= 0.45 * max_h]
    tall_confident = [f for f in tall if f['conf'] >= 0.30]

    hypotheses = []
    seen_texts = set()
    for frag_set, label in ((tall, ""),                # Ana hipotez
                            (tall_confident, "+güven"),  # Güven filtreli
                            (fragments, "+tümü")):       # Filtresiz
        text, conf = _join_fragments(frag_set)
        # En az 4 karakter olsun ve aynı metin tekrar aday olmasın
        if len(text) >= 4 and text not in seen_texts:
            seen_texts.add(text)
            hypotheses.append((text, conf, label))

    return hypotheses


def _character_vote(scored_candidates):
    """
    [YENİ YARDIMCI — KARAKTER SEVİYESİNDE OYLAMA (ENSEMBLE)]

    Farklı stratejiler aynı plakayı çoğunlukla 1 karakter farkla okur:
    strateji A: "06HBY325" (güven 0.81)  ← H hatalı
    strateji B: "06ABY325" (güven 0.51)  ← doğru
    strateji C: "06ABY325" (güven 0.50)  ← doğru

    Tek tek bakınca en güvenli aday A (hatalı!). Ama karakter bazında
    güven-ağırlıklı oylama yapılırsa: pozisyon 2'de A=0.81'e karşı
    A+B=1.01 → 'A' kazanır → "06ABY325" sentezlenir.

    Sadece AYNI YAPIDA (aynı uzunluk + aynı harf/rakam deseni) geçerli
    adaylar oylanır — farklı yapılar karıştırılmaz.

    Parametreler:
        scored_candidates: [(formatted, conf), ...] — geçerli adaylar

    Returns:
        (sentez_metin, grup_max_güven) veya (None, 0)
    """
    if len(scored_candidates) < 2:
        return None, 0.0

    # Yapı imzasına göre grupla: uzunluk + harf/rakam deseni
    groups = {}
    for formatted, conf in scored_candidates:
        clean = formatted.replace(" ", "")
        signature = (len(clean), tuple(c.isalpha() for c in clean))
        groups.setdefault(signature, []).append((clean, conf))

    # En çok üyeli (eşitlikte toplam güveni yüksek) grubu seç
    best_group = max(groups.values(),
                     key=lambda g: (len(g), sum(c for _, c in g)))
    if len(best_group) < 2:
        return None, 0.0

    # Pozisyon pozisyon güven-ağırlıklı oylama
    length = len(best_group[0][0])
    synthesized = []
    for pos in range(length):
        votes = {}
        for clean, conf in best_group:
            votes[clean[pos]] = votes.get(clean[pos], 0.0) + conf
        synthesized.append(max(votes, key=votes.get))

    # Sentezin güveni: grubun en güvenilir üyesi kadar güvenilir kabul
    # edilir (oylama en az en iyi üye kadar iyidir varsayımı)
    max_conf = max(c for _, c in best_group)
    return "".join(synthesized), max_conf


def read_plate_ocr(image_variants, reader):
    """
    [YENİDEN YAZILDI — FORMAT-FARKINDA ÇOK STRATEJİLİ OCR]

    ESKİ KODDAKİ SORUNLAR:
    1. Tek görüntü varyantı üzerinde deniyordu — temizleme adımları
       karaktere zarar verirse geri dönüş yolu yoktu.
    2. En iyi sonuç SADECE güven oranına göre seçiliyordu. OCR bazen
       çöp metne yüksek güven verir.

    YENİ YAKLAŞIM:
    1. BİRDEN FAZLA GÖRÜNTÜ VARYANTI (çağıran belirler):
       "temiz" (tam temizlenmiş), "ham" (eğiklik düzeltilmiş kırpım),
       "sade" (yalnızca 4x büyütülmüş dar kırpım). Her varyant farklı
       hata türlerini telafi eder.
    2. VARSAYILAN EASYOCR PARAMETRELERİ:
       Deneylerle kanıtlandı: düşürülmüş text_threshold/low_text gibi
       "ayarlı" parametreler gürültüyü metin bölgesine karıştırıp
       doğruluğu DÜŞÜRÜYOR ('42AEH738' → '42M738' örneği). Varsayılan
       parametreler + allowlist en isabetli kombinasyon.
    3. STRATEJİLER: renkli + OTSU. Adaptif binarize KALDIRILDI —
       testlerde tutarlı biçimde çöp üretti ('3448', 'B8BY5' gibi).
    4. ÇOK SİNYALLİ SKORLAMA:
       skor = güven
            + geçerlilik bonusu (onarım maliyetiyle azalır: 1.0 - 0.15×maliyet)
            + konsensüs bonusu (aynı sonucu veren her ek strateji: +0.15, en çok +0.30)
            + altdizi bonusu (+0.25: başka geçerli adayı kapsayan daha
              uzun geçerli aday — düşük çözünürlükte karakter DÜŞMESİ
              yaygındır, ortaya karakter EKLENMESİ nadirdir; '42AH738'
              yerine '42AEH738' tercih edilmeli)
    5. KARAKTER OYLAMASI (ensemble): aynı yapıdaki geçerli adaylar
       karakter bazında güven-ağırlıklı oylanır — tek karakter hataları
       çoğunluk kararıyla düzelir (detay _character_vote'ta).

    Parametreler:
        image_variants: [(etiket, görüntü), ...] listesi
        reader: EasyOCR Reader nesnesi

    Returns:
        best_text (str): En iyi ham OCR metni (veya None)
        best_confidence (float): Güven oranı (0.0-1.0)
        best_formatted (str): Formatlanmış plaka metni
        best_valid (bool): Türk plaka yapısal kurallarına uyuyor mu?
        best_score (float): Toplam skor (aday kutular arası karşılaştırma için)
    """
    PLATE_ALLOWLIST = 'ABCDEFGHIJKLMNOPRSTUVYZ0123456789'

    # [DEĞİŞTİ] Varsayılan parametreler — deneyler ayarlı parametrelerin
    # zarar verdiğini gösterdi (gerekçe docstring madde 2)
    OCR_PARAMS = dict(allowlist=PLATE_ALLOWLIST, paragraph=False)

    # ─── BAĞIMSIZ DENEME AŞAMASI ───
    # 5 varyant (temiz/ham/sade/ince/gölge) × 2 strateji (renkli/otsu) =
    # en fazla 10 TAMAMEN BAĞIMSIZ reader.readtext() çağrısı yapılır.
    # Her biri kendi metnini/güvenini üretir, BİRBİRİNİN SONUCUNU GÖRMEZ.
    # Birleştirme/karşılaştırma burada değil, aşağıda (ADIM 1-3'te) TÜM
    # adaylar toplandıktan SONRA tek seferde yapılır — bağımsızlık şart,
    # çünkü varyant AİLESİ konsensüsü (aşağıda) ancak denemeler birbirini
    # etkilemediyse anlamlıdır.
    candidates = []

    for variant_label, variant_image in image_variants:
        if variant_image is None or variant_image.size == 0:
            continue

        # [DEĞİŞTİ] Stratejiler: renkli + OTSU (adaptif binarize kaldırıldı)
        strategies = [("renkli", variant_image)]
        try:
            if len(variant_image.shape) == 3:
                gray = cv2.cvtColor(variant_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = variant_image
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            strategies.append(("otsu", otsu))
        except Exception:
            pass

        for strategy_name, strategy_image in strategies:
            try:
                result = reader.readtext(strategy_image, **OCR_PARAMS)
            except Exception as e:
                print(f"    [OCR] {variant_label}/{strategy_name} hatası: {e}")
                continue

            # [DEĞİŞTİ] merge artık birden fazla birleştirme hipotezi
            # döndürüyor — her biri ayrı aday olarak skorlamaya girer
            for text, conf, hyp_label in merge_ocr_fragments(result):
                candidates.append((text, conf,
                                   f"{variant_label}/{strategy_name}{hyp_label}"))

    if not candidates:
        print("    [OCR] Hiçbir strateji sonuç vermedi!")
        return None, 0.0, "", False, 0.0

    # ─── ADIM 1: Her adayı formatla ve onarım maliyetini al ───
    processed = []
    for text, conf, strategy in candidates:
        formatted, valid, repair_cost = format_turkish_plate_ex(text)
        processed.append({
            'text': text, 'conf': conf, 'strategy': strategy,
            'formatted': formatted, 'valid': valid, 'cost': repair_cost,
        })

    # ─── ADIM 2: Karakter oylaması — geçerli adaylardan sentez üret ───
    valid_pool = [(p['formatted'], p['conf']) for p in processed if p['valid']]
    voted_text, voted_conf = _character_vote(valid_pool)
    if voted_text:
        formatted, valid, repair_cost = format_turkish_plate_ex(voted_text)
        if valid and not any(p['formatted'] == formatted and p['strategy'] == 'oylama'
                             for p in processed):
            processed.append({
                'text': voted_text, 'conf': voted_conf, 'strategy': 'oylama',
                'formatted': formatted, 'valid': valid, 'cost': repair_cost,
            })

    # ─── ADIM 3: Çok sinyalli skorlama ───
    # [DEĞİŞTİ] Konsensüs artık VARYANT AİLESİ bazında sayılır.
    # NEDEN? temiz/ham/sade varyantları hepsi "kalın karakterli" aynı
    # görüntü ailesindendir — düşük çözünürlükte A harfini üçü birden
    # H okur (aynı kök neden = korelasyonlu hata). Bunları bağımsız
    # kanıt saymak sistematik hatayı ödüllendiriyordu. "ince"
    # (inceltilmiş) varyant ayrı ailedir; yalnız FARKLI ailelerin
    # uzlaşması gerçek bağımsız doğrulamadır.
    def variant_family(strategy):
        # "gölge" varyantı da inceltilmiş ailedendir (aynı dilasyon işlemi)
        return "ince" if strategy.startswith(("ince", "gölge")) else "kalın"

    format_families = {}
    for p in processed:
        if p['strategy'] != 'oylama':  # Sentez kendi kaynaklarını saymasın
            format_families.setdefault(p['formatted'], set()).add(
                variant_family(p['strategy']))

    scored = []
    for p in processed:
        score = p['conf']

        # Geçerlilik bonusu — onarım maliyetiyle azalır
        if p['valid']:
            score += max(0.3, 1.0 - 0.15 * p['cost'])
        elif re.match(r'^\d{2} [A-Z]{1,3} \d{2,4}$', p['formatted']):
            score += 0.2

        # Konsensüs bonusu — FARKLI varyant aileleri aynı sonuca vardıysa
        agreement = len(format_families.get(p['formatted'], set())) - 1
        score += min(0.30, 0.15 * max(0, agreement))

        # ─── İNCELTME ÇÖZÜMLEME BONUSU (A↔H) ───  [YENİ]
        # Fiziksel mekanizma TEK YÖNLÜdür: bulanıklık/kalınlaşma A'nın
        # iç boşluğunu doldurup H'ye çevirir; ama inceltme H'nin iki dik
        # çizgisini A'nın üçgen tepesine ÇEVİREMEZ. Dolayısıyla ince
        # varyant yüksek güvenle 'A' okuyorsa ve kalın varyantlar aynı
        # pozisyonda 'H' diyorsa, 'A' okuması fiziksel olarak güvenilir
        # olandır. (Deney: '20 HFB 280' 3 varyantta ısrarcıydı, ince
        # varyant 1.00 güvenle '20 AFB 280' okudu — doğrusu da buydu.)
        if p['valid'] and p['strategy'].startswith('ince') and p['conf'] >= 0.80:
            my_clean = p['formatted'].replace(" ", "")
            for q in processed:
                if q is p or not q['valid'] or q['strategy'].startswith('ince'):
                    continue
                other = q['formatted'].replace(" ", "")
                if len(other) == len(my_clean):
                    diffs = [(a, b) for a, b in zip(my_clean, other) if a != b]
                    if diffs and all(a == 'A' and b == 'H' for a, b in diffs):
                        score += 0.35
                        break

        # Altdizi bonusu — bu geçerli aday, başka bir geçerli adayı
        # altdizi olarak kapsıyorsa (karakter düşmesini telafi)
        # [DEĞİŞTİ] EK KOŞUL: uzun adayın aile desteği, kapsadığı kısa
        # adayınkinden az OLMAMALI. Gerçek hata: gölge varyantı TR bandı
        # kalıntısından sahte 'S' ekleyip '34SN5953' üretti (tek aile);
        # doğru okuma '34N5953' 2 aileden destekliydi ama altdizi bonusu
        # sahte eklemeyi ödüllendirip yanlışı kazandırdı. Karakter
        # düşmesi telafisi ancak uzun okuma en az kısa okuma kadar
        # bağımsız kanıta sahipse güvenilirdir.
        if p['valid']:
            my_clean = p['formatted'].replace(" ", "")
            my_fams = len(format_families.get(p['formatted'], set()))
            for q in processed:
                if q['valid'] and q is not p:
                    other = q['formatted'].replace(" ", "")
                    other_fams = len(format_families.get(q['formatted'], set()))
                    if (len(other) < len(my_clean)
                            and _is_subsequence(other, my_clean)
                            and my_fams >= other_fams):
                        score += 0.25
                        break

        scored.append((score, p))

    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best = scored[0]

    print(f"    [OCR] Sonuçlar (skor = güven + geçerlilik + konsensüs + altdizi):")
    for score, p in scored:
        marker = " ✓" if p is best else ""
        valid_str = "GEÇERLİ" if p['valid'] else "geçersiz"
        print(f"           {p['strategy']:>14}: '{p['text']}' → '{p['formatted']}' "
              f"[{valid_str}, onarım:{p['cost']}] (güven: {p['conf']:.2f}, "
              f"skor: {score:.2f}){marker}")

    # ─── YABANCI PLAKA GÜVENLİK KONTROLÜ ───  [YENİ]
    # SORUN: Türk allowlist'i W/Q/X harflerini içermez (Türk plakasında
    # kullanılmazlar). Alman "WI TJ 473" plakası bu yüzden yanlış okunur
    # ve onarım mekanizması onu zorla geçerli Türk plakasına dönüştürür
    # ('16 TJ 473') → kendinden emin SAHTE kayıt üretilir!
    # ÇÖZÜM: Sonuç "geçerli Türk plakası" çıktıysa, TAM alfabeyle
    # (W/Q/X dahil) bir doğrulama okuması yap. Ana metin satırında
    # yeterli güvenle W/Q/X görülüyorsa plaka Türk OLAMAZ → YABANCI
    # olarak işaretle ve Türk plakası diye kaydetme.
    # (Deney: Alman plakada tam alfabe 'WI'=%92, TR alfabe 'HI'=%65)
    if best['valid']:
        FULL_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # [DEĞİŞTİ] Kontrol artık TÜM varyantlarda yapılır. Tek varyant
        # yanıltıcıydı: kazanan varyant Alman plakayı 'W6T1277' okurken
        # 'sade' varyantı doğru 'WI TJ473' okuyordu. Her varyantın ana
        # metin satırı (yükseklik + güven filtresiyle) ayrı çıkarılır.
        foreign_readings = []  # (varyant, boşluklu, bitişik, güven)
        for lbl, img in image_variants:
            try:
                fr = reader.readtext(img, allowlist=FULL_ALPHABET,
                                     paragraph=False)
            except Exception:
                continue
            frags = [(bbox, t.replace(' ', '').upper(), c)
                     for bbox, t, c in fr if t.strip()]
            if not frags:
                continue
            heights = [max(p[1] for p in b) - min(p[1] for p in b)
                       for b, _, _ in frags]
            h_max = max(heights)
            main = [(b, t, c) for (b, t, c), h in zip(frags, heights)
                    if h >= 0.45 * h_max and c >= 0.25]
            if not main:
                continue
            main.sort(key=lambda f: min(p[0] for p in f[0]))
            joined = ''.join(t for _, t, _ in main)
            spaced = ' '.join(t for _, t, _ in main)
            conf = (sum(c * len(t) for _, t, c in main)
                    / max(1, sum(len(t) for _, t, _ in main)))
            foreign_readings.append((lbl, spaced, joined, conf))

        # Yabancı kanıtı: W/Q/X EN AZ 2 varyantta görülmeli.
        # [DEĞİŞTİ] Tek varyant istisnası (≥0.60 güven) kaldırıldı —
        # gerçek hata: Türk '06 AB 8655' plakasında tek varyant A'yı W
        # okudu (0.67) ve plaka yanlışlıkla YABANCI işaretlendi. Gerçek
        # yabancı plakada W/Q/X birden çok varyantta tutarlı görünür
        # (Alman plaka deneyi: 4/5 varyant); tek varyantlık W/Q/X ise
        # hayalet okumadır.
        qwx_hits = [(lbl, c) for lbl, _, t, c in foreign_readings
                    if any(ch in t for ch in 'QWX')]
        is_foreign = len(qwx_hits) >= 2

        if is_foreign:
            # Metin seçimi: dilasyonsuz (kalın) varyantlar öncelikli —
            # ince/gölge varyantlarındaki dilasyon, AB plakasının mavi
            # bandından hayalet harf üretebiliyor ('G' gibi).
            kalin_pool = [r for r in foreign_readings
                          if variant_family(r[0]) == 'kalın'
                          and any(ch in r[2] for ch in 'QWX')]
            pool = kalin_pool or [r for r in foreign_readings
                                  if any(ch in r[2] for ch in 'QWX')]
            pool.sort(key=lambda r: r[3], reverse=True)
            _, spaced, joined, fconf = pool[0]
            print(f"    [YABANCI] Tam alfabe okuması '{spaced}' "
                  f"(güven: {fconf:.2f}, {len(qwx_hits)} varyantta W/Q/X) → "
                  f"Türk plakası DEĞİL, kayıt Türk formatına zorlanmadı!")
            # Yabancı okuma metnini göster, geçersiz işaretle
            # (valid=False → CSV kalite kapısı bunu loglamaz)
            return joined, fconf, f"{spaced} [YABANCI]", False, 0.0

    # ─── TEK KAYNAK GÜVENCESİ ───  [YENİ]
    # Kazanan okuma yalnızca TEK varyant ailesinden destek alıyorsa ve
    # güveni orta seviyedeyse (< 0.60), sonuç "şüpheli" bandına indirilir
    # (güven 0.30 olarak raporlanır → özet tablosunda '?' işareti alır,
    # CSV kalite kapısından geçemez).
    # NEDEN? Gölgeli/bozuk plakalarda tek bir varyant yapısal olarak
    # geçerli ama YANLIŞ okuma üretebilir (gerçek hata: gölgedeki
    # '34 JA 8191' plakası tek kaynaktan '34 JLB 191' okundu, %46
    # güvenle kalite kapısını geçip CSV'ye yanlış kayıt olarak girdi).
    # Bağımsız doğrulaması olmayan orta güvenli okuma iddia edilmemeli.
    if best['valid']:
        family_support = len(format_families.get(best['formatted'], set()))
        if family_support <= 1 and best['conf'] < 0.60:
            print(f"    [ŞÜPHELİ] '{best['formatted']}' tek varyant ailesinden "
                  f"(güven: {best['conf']:.2f} < 0.60) → şüpheli olarak işaretlendi")
            return best['text'], 0.30, best['formatted'], True, best_score

    # ─── YABANCI ETİKETİ STANDARDİZASYONU ───  [YENİ]
    # Türk plaka formatına uymayan her OKUNABİLİR sonuç tutarlı biçimde
    # [YABANCI] etiketi taşır — W/Q/X kanıtı olsun (yukarıdaki kontrol)
    # ya da olmasın ('SNIP3R' gibi süs/yabancı plakalar W/Q/X içermez ama
    # Türk plakası da değildir). Ayrım güvenle yapılır:
    #   - geçersiz + güven ≥ 0.40 → net okunmuş ama Türk formatı değil
    #     = YABANCI
    #   - geçersiz + güven < 0.40 → muhtemelen kötü okunmuş bir plaka
    #     = "okunamadı" (yabancı İDDİA EDİLMEZ, Türk plakası olabilir)
    if not best['valid'] and best['conf'] >= 0.40:
        print(f"    [YABANCI] '{best['formatted']}' Türk formatına uymuyor "
              f"ama net okundu (güven: {best['conf']:.2f}) → YABANCI etiketi")
        return (best['text'], best['conf'],
                f"{best['formatted']} [YABANCI]", False, best_score)

    return best['text'], best['conf'], best['formatted'], best['valid'], best_score


def _is_subsequence(short, long):
    """short'un tüm karakterleri long içinde aynı sırayla geçiyor mu?"""
    it = iter(long)
    return all(ch in it for ch in short)


# =============================================================================
# BÖLÜM 8: TÜRK PLAKA FORMAT DÜZELTMESİ
# =============================================================================

# Harf → Rakam dönüşüm tablosu (OCR'ın karıştırdığı benzer şekiller)
# Modül seviyesine taşındı — hem format hem doğrulama fonksiyonları kullanıyor
LETTER_TO_DIGIT = {
    'O': '0', 'Q': '0',  # O ve Q → 0
    'I': '1', 'L': '1',  # I ve L → 1
    'Z': '2',             # Z → 2
    'S': '5',             # S → 5
    'G': '6',             # G → 6
    'T': '7',             # T → 7
    'B': '8',             # B → 8
}

# Rakam → Harf dönüşüm tablosu
DIGIT_TO_LETTER = {
    '0': 'O',  # 0 → O
    '1': 'I',  # 1 → I
    '2': 'Z',  # 2 → Z
    '5': 'S',  # 5 → S
    '6': 'G',  # 6 → G
    '8': 'B',  # 8 → B
}


def is_valid_turkish_plate(text):
    """
    [YENİ FONKSİYON — YAPISAL PLAKA DOĞRULAMA]

    Eski kod sadece şekil kontrolü yapıyordu: "2 rakam + 1-3 harf + 2-4 rakam".
    Ama Türk plaka sisteminde harf sayısı ile rakam sayısı BİRBİRİNE BAĞLIDIR:

    ┌──────────────┬───────────────┬─────────────────┐
    │ Harf sayısı  │ Rakam sayısı  │ Örnek           │
    ├──────────────┼───────────────┼─────────────────┤
    │ 1 harf       │ 4 rakam       │ 34 N 5953       │
    │ 2 harf       │ 3 veya 4      │ 42 AH 738       │
    │ 3 harf       │ 2 veya 3      │ 06 ABY 325      │
    └──────────────┴───────────────┴─────────────────┘

    Bu kural OCR artefaktlarını otomatik yakalar:
    "06 ABY 3251" → 3 harf + 4 rakam = GEÇERSİZ → sondaki artefakt "1"
    kırpılınca "06 ABY 325" = GEÇERLİ. (Gerçek test hatasından!)

    Ek olarak il kodu 01-81 aralığında olmalıdır (Türkiye'de 81 il var).
    "95 ABC 123" gibi okumalar kesinlikle OCR hatasıdır.

    Parametreler:
        text (str): Boşluksuz aday metin (örn. "06ABY325")

    Returns:
        bool: Tüm yapısal kurallara uyuyor mu?
    """
    m = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', text)
    if not m:
        return False

    # İl kodu kontrolü: 01-81
    il_code = int(m.group(1))
    if not (1 <= il_code <= 81):
        return False

    # Harf sayısı ↔ rakam sayısı ilişkisi
    letter_count = len(m.group(2))
    digit_count = len(m.group(3))
    valid_combos = {(1, 4), (2, 3), (2, 4), (3, 2), (3, 3)}
    return (letter_count, digit_count) in valid_combos


def _positional_fix(text):
    """
    [YENİ YARDIMCI — POZİSYON BAZLI KARAKTER DÜZELTME]

    Türk plakasının bölge yapısına göre karışan karakterleri düzeltir:
    - İl kodu bölgesi (ilk 2 karakter): harf görünüyorsa rakama çevir
      (örn. "O6ABY325" → "06ABY325")
    - Numara bölgesi (sondaki rakam bloğu): harf karıştıysa rakama çevir
      (örn. "34ABC7B8" → "34ABC788")
      GÜVENLİK: sondan zaten 2+ gerçek rakam varsa harf çevirmeyi durdur —
      yoksa "34ABT738"deki gerçek T harfi 7'ye çevrilirdi!
    - Orta bölge (harf serisi): rakam karıştıysa harfe çevir
      (örn. "340BC123" → "34OBC123")

    Parametreler:
        text (str): Boşluksuz ham metin

    Returns:
        str: Düzeltilmiş metin
    """
    if len(text) < 5:
        return text

    chars = list(text)

    # 1) İl kodu bölgesi: ilk 2 karakter rakam olmalı
    for i in range(2):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # 2) Numara bölgesi: sondan geriye doğru rakam bloğunu belirle
    j = len(chars) - 1
    digit_count = 0
    while j >= 2 and digit_count < 4:
        c = chars[j]
        if c.isdigit():
            digit_count += 1
            j -= 1
        elif c in LETTER_TO_DIGIT and digit_count < 2:
            # Sadece rakam bloğu henüz 2'den kısaysa harf→rakam çevir.
            # (2+ rakam zaten varsa bu harf muhtemelen serinin son harfidir)
            chars[j] = LETTER_TO_DIGIT[c]
            digit_count += 1
            j -= 1
        else:
            break

    # 3) Orta bölge (2..j): harf serisi — rakamları harfe çevir
    for k in range(2, j + 1):
        if chars[k] in DIGIT_TO_LETTER:
            chars[k] = DIGIT_TO_LETTER[chars[k]]

    return "".join(chars)


def format_turkish_plate_ex(raw_text):
    """
    [YENİDEN YAZILDI] OCR çıktısını standart Türk plaka formatına düzeltir,
    YAPISAL GEÇERLİLİĞİNİ doğrular ve ONARIM MALİYETİNİ raporlar.

    ESKİ KODDAN FARKLAR:
    1. GEÇERLİLİK BİLGİSİ DÖNER — OCR skorlaması geçerli plakaları
       güven farkına rağmen öne çekebilsin diye.
    2. ARTEFAKT KIRPMA: Plaka çerçevesi/vida/gölge kaynaklı fazla
       karakterleri baştan ve sondan 0-2 karakter kırparak dener.
       Örn: "06ABY3251" (çerçeve kenarı '1' okundu) → son karakteri
       kırp → "06ABY325" → yapısal olarak geçerli → kabul!
    3. ONARIM MALİYETİ: kaç karakterin kırpıldığı + kaç karakterin
       dönüştürüldüğü sayılır ve döndürülür.
       NEDEN KRİTİK? İlk sürümde agresif onarım, çöp okumaları bile
       "geçerli" plakaya dönüştürüp yüksek skor almalarını sağlıyordu
       (gerçek hata: '3LM5953' onarımla '31 M 5953' olup doğru okuma
       '34N5953'ü yendi!). Skorlama artık onarım maliyetiyle orantılı
       ceza keser: onarımsız geçerli okuma her zaman önde olur.
    4. TÜM ONARIM YOLLARI DENENİR, EN UCUZU SEÇİLİR:
       Eşit maliyette dönüşümü az olan tercih edilir — kırpma kenar
       artefaktı atar (yaygın ve güvenli), dönüşüm ise karakter kanıtını
       değiştirir (daha riskli).

    Parametreler:
        raw_text (str): OCR'dan gelen ham metin

    Returns:
        formatted (str): Formatlanmış plaka metni ("XX YYY ZZZZ")
        valid (bool): Türk plaka yapısal kurallarına uyuyor mu?
        repair_cost (int): Onarım maliyeti (0 = hiç onarım gerekmedi)
    """
    if not raw_text:
        return "", False, 0

    # Temizle: harf/rakam dışındaki her şeyi at + büyük harfe çevir
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    if len(text) < 5:
        return text, False, 0

    # ─── TÜM ONARIM YOLLARINI TOPLA, EN UCUZUNU SEÇ ───
    # Kırpma kombinasyonları (0-2 baştan, 0-2 sondan) × (ham, düzeltilmiş)
    n = len(text)
    valid_options = []   # (toplam_maliyet, dönüşüm_sayısı, düzeltilmiş_metin)
    seen = set()
    MAX_REPAIR_COST = 3  # Bundan pahalı onarımlar güvenilmez — kabul etme

    for lead in range(0, 3):
        for trail in range(0, 3):
            if n - lead - trail < 5:
                continue
            candidate = text[lead:n - trail] if trail else text[lead:]
            fixed = _positional_fix(candidate)
            # Dönüşüm sayısı: pozisyon düzeltmesinin değiştirdiği karakterler
            conversions = sum(1 for a, b in zip(candidate, fixed) if a != b)

            for variant, conv_count in ((candidate, 0), (fixed, conversions)):
                if variant in seen:
                    continue
                seen.add(variant)
                cost = lead + trail + conv_count
                if cost > MAX_REPAIR_COST:
                    continue
                if is_valid_turkish_plate(variant):
                    valid_options.append((cost, conv_count, variant))

    if valid_options:
        # En düşük maliyet; eşitlikte en az dönüşüm (kırpma > dönüşüm tercihi)
        valid_options.sort(key=lambda o: (o[0], o[1]))
        cost, _, best = valid_options[0]
        m = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', best)
        return f"{m.group(1)} {m.group(2)} {m.group(3)}", True, cost

    # ─── GEÇERLİ SONUÇ YOK — EN İYİ ÇABA FORMATLAMASI ───
    # Yapısal kural geçmese bile şekil uyuyorsa boşluklu formatla döndür
    # (kullanıcı çıktıda yine de okuyabilsin)
    fixed = _positional_fix(text)
    m = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', fixed)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}", False, 0

    return text, False, 0


def format_turkish_plate(raw_text):
    """Geriye dönük uyumlu sarmalayıcı — (metin, geçerli_mi) döndürür."""
    formatted, valid, _ = format_turkish_plate_ex(raw_text)
    return formatted, valid


# =============================================================================
# BÖLÜM 8.5: ÇOK GEÇİŞLİ PLAKA TESPİTİ VE EĞİKLİK DÜZELTME  [YENİ]
# =============================================================================

def detect_plate_boxes(model, frame, device):
    """
    [TESPİT HATALARININ ANA ÇÖZÜMÜ]

    ESKİ SORUN:
    model.predict() varsayılan imgsz=640 ile çalışıyordu. 2048x1536 gibi
    yüksek çözünürlüklü fotoğraflar YOLO'ya girmeden önce 640px'e
    küçültülüyordu → plaka görüntüde 15-25 piksele düşüyordu → model
    plakayı HİÇ tespit edemiyordu. Test setindeki 6 görüntünün 3'ünde
    başarısızlığın kök nedeni buydu (OCR değil, tespit aşaması!).

    YENİ YAKLAŞIM — KADEMELİ ÇOK GEÇİŞLİ TESPİT:
    1. Geçiş: imgsz=1280, conf=0.25 → normal durumlar (hızlı ve güvenilir)
    2. Geçiş: imgsz=1920, conf=0.10 → küçük/uzak plakalar
    3. Geçiş: imgsz=640,  conf=0.03 → modelin EĞİTİM çözünürlüğü + çok
       düşük eşik. Deneyler gösterdi ki model eğitildiği 640px'te bazı
       plakaları düşük güvenle de olsa görebiliyorken, yüksek
       çözünürlüklerde hiç göremiyor (ölçek genellemesi zayıf).
       Düşük eşiğin riski yok: geometri filtresi + OCR yapısal
       doğrulaması yanlış tespitleri zaten eler.

    [DEĞİŞTİ] ERKEN ÇIKIŞ KALDIRILDI — artık TÜM geçişler çalışır ve
    kutular birleştirilir (IoU tekilleştirmesi ile). NEDEN? Gerçek hata:
    yakın çekim SNIP3R fotoğrafında 1. geçiş (1280) plakayı değil,
    çerçevenin "#FeelTheMPower" yazı şeridini kutuladı ve erken çıkış
    yüzünden 640 geçişine hiç sıra gelmedi — oysa gerçek plaka (görüntünün
    %73'ü genişliğinde, çok büyük) yalnızca küçük imgsz'de bulunuyordu
    (320px'te güven 0.92). Büyük/yakın plakalar küçük ölçekte, küçük/uzak
    plakalar büyük ölçekte görünür; ikisini de yakalamak için geçişlerin
    tamamı gerekir. Yanlış kutular (yazı şeridi gibi) güven sıralamasında
    geriye düşer ve OCR yapısal doğrulamasında zaten elenir.

    Parametreler:
        model: YOLO model nesnesi
        frame (np.ndarray): BGR formatında tam kare görüntü
        device (str): "0" (GPU) veya "cpu"

    Returns:
        valid_boxes (list): [(x1, y1, x2, y2, conf, ar), ...] güvene göre sıralı
        pass_no (int): Kaçıncı geçişte bulunduğu (istatistik için)
    """
    # Plaka geometri filtresi sınırları (eski koddan taşındı)
    PLATE_AR_MIN = 1.5   # Minimum en/boy oranı (eğik açı toleransı)
    PLATE_AR_MAX = 7.5   # Maximum en/boy oranı
    MIN_WIDTH = 30       # Minimum plaka genişliği (piksel)
    MIN_HEIGHT = 10      # Minimum plaka yüksekliği (piksel)

    # Kademeli geçiş planı: önce ucuz/hızlı, gerekirse alternatif ölçekler
    # [DEĞİŞTİ] 3. geçiş 2560+TTA yerine 640+conf=0.03 yapıldı —
    # deneyler 2560'ın hiçbir ek plaka bulamadığını, 640'ın (eğitim
    # çözünürlüğü) ise düşük eşikte ek plakalar yakaladığını gösterdi
    detection_passes = [
        {"imgsz": 1280, "conf": 0.25},
        {"imgsz": 1920, "conf": 0.10},
        {"imgsz": 640,  "conf": 0.03},
    ]

    all_boxes = []  # tüm geçişlerden toplanan aday kutular
    first_hit_pass = 0

    for pass_no, params in enumerate(detection_passes, 1):
        results = model.predict(source=frame, device=device, verbose=False, **params)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"    [TESPİT] Geçiş {pass_no} (imgsz={params['imgsz']}, "
                  f"conf={params['conf']}): tespit yok")
            continue

        pass_count = 0
        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = map(int, boxes[i].xyxy[0])
            bw, bh = bx2 - bx1, by2 - by1
            bconf = float(boxes[i].conf[0])

            if bh <= 0 or bw <= 0:
                continue

            ar = bw / bh
            if (PLATE_AR_MIN <= ar <= PLATE_AR_MAX and
                    bw >= MIN_WIDTH and bh >= MIN_HEIGHT):
                all_boxes.append((bx1, by1, bx2, by2, bconf, ar))
                pass_count += 1

        if pass_count and not first_hit_pass:
            first_hit_pass = pass_no
        print(f"    [TESPİT] Geçiş {pass_no} (imgsz={params['imgsz']}): "
              f"{pass_count} geçerli kutu")

    if not all_boxes:
        return [], len(detection_passes)

    # ─── KUTU TEKİLLEŞTİRME ───
    # Aynı plaka farklı ölçek geçişlerinde tekrar bulunur; örtüşen
    # kutulardan en güvenilir olan tutulur. İki kutu şu durumda aynı
    # sayılır: IoU > 0.5 VEYA küçük kutunun %85'i büyüğün içinde
    # (iç içe tespit — sıkı kutu vs gevşek kutu).
    def _overlap(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0, 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        iou = inter / (area_a + area_b - inter)
        containment = inter / min(area_a, area_b)
        return iou, containment

    all_boxes.sort(key=lambda b: b[4], reverse=True)  # en güvenilir önce
    valid_boxes = []
    for box in all_boxes:
        duplicate = False
        for kept in valid_boxes:
            iou, cont = _overlap(box, kept)
            if iou > 0.5 or cont > 0.85:
                duplicate = True
                break
        if not duplicate:
            valid_boxes.append(box)

    print(f"    [TESPİT] Birleştirme: {len(all_boxes)} kutu → "
          f"{len(valid_boxes)} tekil plaka adayı")
    return valid_boxes, first_hit_pass


def detect_plate_classical(frame, max_candidates=3):
    """
    [YENİ FONKSİYON — KLASİK GÖRÜNTÜ İŞLEME İLE YEDEK PLAKA TESPİTİ]

    NEDEN GEREKLİ?
    YOLO modeli (best.pt) bazı açılardan/koşullardan çekilen plakaları
    conf=0.01'de bile HİÇ tespit edemiyor (deneylerle doğrulandı —
    model bu tür örneklerle yeterince eğitilmemiş). Model yeniden
    eğitilene kadar, derin öğrenme gerektirmeyen klasik bir plaka
    bulucu son çare olarak devreye girer.

    ALGORİTMA (klasik ANPR ön-tespit yaklaşımı):
    1. Görüntüyü ~1200px çalışma genişliğine ölçekle (hız/detay dengesi)
    2. BLACKHAT morfolojisi: parlak zemin üzerindeki koyu yapıları
       (plaka harfleri!) vurgular — plakanın imzası budur
    3. Parlaklık maskesi: plaka zemini beyazdır → OTSU ile parlak
       bölgeleri maskele
    4. X-yönlü Sobel gradyanı: harflerin dikey kenarları yan yana
       dizilince güçlü yatay gradyan deseni oluşur
    5. Gauss bulanıklaştırma + yatay kapama (closing): harf kenarlarını
       tek bitişik blok haline getir
    6. Kontur analizi: en/boy oranı ve boyut plaka geometrisine uyan,
       zemini yeterince parlak bölgeleri skorla

    AR ALT SINIRI NEDEN 1.3 (2.0 DEĞİL)?
    Morfolojik kapama plaka blobunu bazen üstündeki/altındaki tampon
    çizgisiyle birleştirir → blob plakadan daha kare görünür (deneyde
    AR=1.5 ölçüldü). Gevşek sınır kabul edilir çünkü asıl doğrulamayı
    OCR + yapısal plaka kuralları yapar — yanlış adaylar orada elenir.

    Parametreler:
        frame (np.ndarray): BGR formatında tam kare görüntü
        max_candidates (int): Döndürülecek en iyi aday sayısı

    Returns:
        candidates (list): [(x1, y1, x2, y2, skor, ar), ...] skora göre sıralı
                           (koordinatlar orijinal görüntü ölçeğinde)
    """
    H, W = frame.shape[:2]

    # Çalışma ölçeği: ~1200px genişlik (küçük görüntüler en fazla 2.5x büyür)
    scale = min(1200.0 / W, 2.5)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]

    # 1) Blackhat: kapanış(görüntü) - görüntü → parlak zemindeki koyu
    #    detayları (harfleri) beyaz olarak öne çıkarır
    #    Kernel (25,7): yatay uzun — plaka metin bloğunun şekline uygun
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)

    # 2) Parlak bölgeler maskesi (plaka zemini beyaz/açık renk)
    square_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kern)
    _, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3) X-yönlü gradyan — harf dikey kenarlarının yoğun olduğu bölgeler
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    mn, mx = grad_x.min(), grad_x.max()
    grad_x = (255 * ((grad_x - mn) / (mx - mn + 1e-6))).astype("uint8")

    # 4) Bulanıklaştır + yatay kapama → harfleri tek blok yap + eşikle
    grad_x = cv2.GaussianBlur(grad_x, (7, 7), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kern)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 5) Gürültü temizliği + parlaklık maskesiyle kesişim
    #    (koyu zeminli sahte bloklar elenir — plaka zemini parlak olmalı)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    # 6) Kontur analizi ve skorlama
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h else 0
        # Geometri filtresi (gevşek — kesin karar OCR doğrulamasında)
        if not (1.3 <= ar <= 8.0):
            continue
        if w < 50 or h < 12 or w > 0.6 * iw or h > 0.3 * ih:
            continue
        # Zemin parlaklığı: bölgenin en az %35'i parlak olmalı
        white_ratio = light[y:y+h, x:x+w].mean() / 255.0
        if white_ratio < 0.35:
            continue
        # Skor = parlaklık × metin enerjisi (blackhat yoğunluğu) × alan
        # Üçü birden yüksekse: parlak zeminli, koyu metinli, büyükçe bölge
        text_energy = blackhat[y:y+h, x:x+w].mean() / 255.0
        score = white_ratio * text_energy * w * h
        scored.append((score, x, y, w, h))

    scored.sort(key=lambda t: t[0], reverse=True)

    # Koordinatları orijinal ölçeğe geri çevir
    candidates = []
    for score, x, y, w, h in scored[:max_candidates]:
        x1, y1 = int(x / scale), int(y / scale)
        x2, y2 = int((x + w) / scale), int((y + h) / scale)
        ar = (x2 - x1) / max(1, (y2 - y1))
        candidates.append((x1, y1, x2, y2, score, ar))

    return candidates


def rectify_plate(plate_crop):
    """
    [EĞİK PLAKALARI DÜZLEŞTİRME]

    NEDEN GEREKLİ?
    Araç fotoğrafları nadiren tam karşıdan çekilir. Eğik plakalarda
    karakterler yatık durur ve OCR doğruluğu ciddi düşer. Küçük bir
    döndürme düzeltmesi (deskew) bile OCR başarısını belirgin artırır.

    NASIL ÇALIŞIR:
    1. OTSU eşikleme ile parlak plaka gövdesini maskele
       (plaka beyaz/açık renkli olduğu için en büyük parlak bölgedir)
    2. En büyük konturun minAreaRect'i ile eğim açısını ölç
       (minAreaRect: konturu çevreleyen DÖNDÜRÜLMÜŞ minimum dikdörtgen —
        boundingRect'ten farkı, açıyı da vermesidir)
    3. Açı 1°-30° aralığındaysa görüntüyü ters yönde döndürerek düzelt
       - <1°: zaten düz, dokunma (gereksiz interpolasyon kaybı olmasın)
       - >30°: muhtemelen yanlış kontur ölçümü, güvenme

    Parametreler:
        plate_crop (np.ndarray): BGR formatında plaka kırpımı

    Returns:
        result (np.ndarray): Düzleştirilmiş (veya orijinal) görüntü
        was_rotated (bool): Döndürme uygulandı mı?
    """
    h, w = plate_crop.shape[:2]
    # Çok küçük kırpımlarda güvenilir açı ölçülemez
    if h < 15 or w < 40:
        return plate_crop, False

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # THRESH_BINARY + OTSU: parlak bölgeler (plaka gövdesi) beyaz olur
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop, False

    largest = max(contours, key=cv2.contourArea)
    # Plaka gövdesi kırpımın en az %30'unu kaplamalı — yoksa yanlış kontur
    if cv2.contourArea(largest) < 0.30 * h * w:
        return plate_crop, False

    # minAreaRect açı kuralı: OpenCV açıyı [-90, 0) aralığında döndürür,
    # genişlik/yükseklik hangisinin uzun olduğuna göre yorumlanmalı
    (_, _), (rw, rh), angle = cv2.minAreaRect(largest)
    if rw < rh:
        angle += 90
    if angle > 45:
        angle -= 90

    # Anlamlı ama güvenilir aralıktaki açılarda düzelt
    if abs(angle) < 1.0 or abs(angle) > 30.0:
        return plate_crop, False

    # Görüntü merkezinde ters yönde döndür
    # BORDER_REPLICATE: döndürme sonrası köşe boşluklarını kenar pikselleriyle
    # doldurur — siyah üçgenler OCR'ı yanıltmasın diye
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(plate_crop, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print(f"    [EĞİM] {angle:.1f}° eğiklik düzeltildi")
    return rotated, True


# =============================================================================
# BÖLÜM 9: TEK GÖRÜNTÜ İŞLEME PİPELINE'I
# =============================================================================

def process_plate_candidate(frame, box, reader, output_dir, stem, save_debug, tag=""):
    """
    [YENİ FONKSİYON — TEK ADAY KUTUNUN UÇTAN UCA İŞLENMESİ]

    Tespit edilen tek bir aday kutuyu işler:
    kırpım → eğiklik düzeltme → temizleme → 3 görüntü varyantı → OCR.

    NEDEN AYRI FONKSİYON?
    Tespit aşaması artık birden fazla aday kutu döndürebiliyor (özellikle
    klasik CV yedek dedektöründe ilk aday plaka olmayabilir). Her aday bu
    fonksiyonla bağımsız işlenir; çağıran taraf yapısal olarak geçerli
    plaka veren ilk adayı kabul eder.

    ÜÇ GÖRÜNTÜ VARYANTI — NEDEN?
    Deneyler her varyantın farklı görüntülerde kazandığını gösterdi:
    - "sade": paysız dar kırpım + 4x büyütme (eski pipeline davranışı).
      Kırpım payı bazı görüntülerde çerçeve/tampon gölgelerini OCR'a
      sokup sahte karakter okutuyordu — sade varyant bundan etkilenmez.
    - "temiz": sticker/bant/altlık temizliği + iyileştirme. Kirli
      plakalarda en iyi sonucu verir.
    - "ham": kırpım paylı + eğiklik düzeltilmiş + iyileştirilmiş.
      Temizleme adımları karaktere zarar verdiyse telafi eder,
      eğik plakalarda kazanır.

    Parametreler:
        frame (np.ndarray): Tam kare BGR görüntü
        box (tuple): (x1, y1, x2, y2, güven, ar) aday kutu
        reader: EasyOCR Reader nesnesi
        output_dir (str): Debug klasörü (None = kaydetme)
        stem (str): Debug dosya adı kökü
        save_debug (bool): Ara görüntüleri kaydet?
        tag (str): Debug dosya adlarına eklenen aday etiketi

    Returns:
        result (dict): plate_text, confidence, formatted, valid, score, steps
                       veya None (kırpım geçersizse)
    """
    x1, y1, x2, y2, detect_conf, ar = box
    steps = []

    # ─── KIRPIM PAYI (PADDING) ───
    # Eğiklik düzeltme için plaka sınırının görünmesi gerekir; kutu her
    # yönde küçük bir payla genişletilir (yatay %4, dikey %10).
    # Paysız "dar" kırpım da ayrıca tutulur — sade varyantın temeli.
    fh, fw = frame.shape[:2]
    pad_x = max(2, int((x2 - x1) * 0.04))
    pad_y = max(2, int((y2 - y1) * 0.10))
    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)

    tight_crop = frame[y1:y2, x1:x2]      # Paysız dar kırpım
    padded_crop = frame[py1:py2, px1:px2]  # Paylı kırpım

    if tight_crop.size == 0 or padded_crop.size == 0:
        return None

    if save_debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_01_original_crop.jpg"), tight_crop)

    # ─── EĞİKLİK DÜZELTME (DESKEW) ───  [YENİ]
    # Paylı kırpımda yapılır (plaka sınırı görünür olmalı ki açı ölçülsün)
    rectified, was_rotated = rectify_plate(padded_crop)
    if was_rotated:
        steps.append("Eğiklik düzeltildi")
        if save_debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_01b_rectified.jpg"), rectified)

    # ─── TEMİZLEME ZİNCİRİ (dar kırpım üzerinde) ───
    current = tight_crop.copy()

    print(f"  [2/7] Mavi TR bandı kontrolü...")
    current, blue_found, blue_width = detect_blue_band(current)
    if blue_found:
        steps.append(f"Mavi bant temizlendi ({blue_width}px)")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_02_no_blue.jpg"), current)

    print(f"  [3/7] Turuncu sticker kontrolü...")
    current, orange_found = detect_and_remove_orange_sticker(current)
    if orange_found:
        steps.append("Turuncu sticker temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_03_no_orange.jpg"), current)

    print(f"  [4/7] Siyah sticker kontrolü...")
    current, black_found = detect_and_remove_black_sticker(current)
    if black_found:
        steps.append("Siyah sticker temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_04_no_black.jpg"), current)

    print(f"  [5/7] Plaka altlığı/çerçeve kontrolü...")
    current = remove_plate_frame_and_holder(current)
    steps.append("Altlık/çerçeve temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_05_no_frame.jpg"), current)

    # ─── BEŞ GÖRÜNTÜ VARYANTI HAZIRLA ───  [DEĞİŞTİ]
    # Aşağıdaki 5 varyant (temiz/ham/sade/ince/gölge) burada sadece
    # ÜRETİLİR — her biri read_plate_ocr() içinde BİRBİRİNDEN BAĞIMSIZ
    # olarak OCR'a sokulacak (bkz. read_plate_ocr başındaki not).
    print(f"  [6/7] Görüntü iyileştirme (4 varyant hazırlanıyor)...")
    enhanced_clean = enhance_plate_for_ocr(current)      # "temiz"
    enhanced_raw = enhance_plate_for_ocr(rectified)      # "ham"
    # "sade": eski pipeline'ın birebir davranışı — 4x bikübik büyütme,
    # başka hiçbir işlem yok (bazı plakalarda en isabetli okuma bu!)
    plain_4x = cv2.resize(tight_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # "ince": gri tonlama + büyütme + DİLASYON (karakter inceltme)  [YENİ]
    # NEDEN? Düşük çözünürlüklü plakalarda bulanıklık karakter çizgilerini
    # kalınlaştırıp birbirine yapıştırır — A harfinin üçgen iç boşluğu
    # dolunca OCR onu H okur (gerçek hata: '20 AFB 280' → '20 HFB 280').
    # Gri görüntüde dilasyon parlak (beyaz) alanı genişletir = siyah
    # karakter çizgilerini İNCELTİR → yapışan çizgiler ayrışır, A'nın iç
    # boşluğu geri açılır. Deney sonucu: 'HFB' (0.60) → 'AFB' (1.00)!
    # Hedef yükseklik ~270px: inceltme deneyde 6x büyütmede doğrulandı,
    # ölçek kırpım boyutuna göre uyarlanır.
    gray_crop = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
    thin_scale = min(max(270.0 / max(gray_crop.shape[0], 1), 1.0), 8.0)
    gray_up = cv2.resize(gray_crop, None, fx=thin_scale, fy=thin_scale,
                         interpolation=cv2.INTER_CUBIC)
    thinned = cv2.dilate(gray_up, np.ones((3, 3), np.uint8), iterations=2)

    # "gölge": AYDINLATMA DÜZLEŞTİRME + büyütme + inceltme  [YENİ]
    # NEDEN? Ağaç gölgesi gibi benekli/dengesiz aydınlatma plakayı
    # kısmen karartır — global ve CLAHE kontrast bile yetmez, OCR hiç
    # okuyamaz (gerçek hata: arabalar.jpeg'deki '34 JA 8191').
    # ÇÖZÜM: Arka plan aydınlatmasını morfolojik KAPAMA ile kestir
    # (51x51'e denk gelen 31x31 kernel karakterleri siler, geriye sadece
    # yavaş değişen aydınlatma haritası kalır) ve görüntüyü bu haritaya
    # BÖL → gölge deseni sadeleşir, karakterler eşit kontrasta gelir.
    # Deney: gölgeli plaka hiç okunamazken bu varyantla '34JL0191'
    # (güven 0.49) seviyesine çıktı.
    # Kapama (dilate+erode) büyük 31x31 çekirdekle karakterleri SİLER,
    # geriye yalnızca kaba/yavaş-değişen aydınlatma haritası kalır
    # (flat_bg = "bu bölge genel olarak ne kadar parlak"). Orijinali bu
    # haritaya BÖLMEK, her pikseli KENDİ yerel referansına göre
    # değerlendirir → gölgedeki koyu zemin ile aydınlıktaki koyu zemin
    # bölme sonrası aynı göreli değere gelir, gölge deseni pratikte iptal olur.
    flat_bg = cv2.morphologyEx(gray_crop, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    flat = cv2.divide(gray_crop, flat_bg, scale=255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)  # bölme sonrası dar kalan kontrastı 0-255'e yay
    golge_scale = min(max(160.0 / max(gray_crop.shape[0], 1), 1.0), 8.0)
    golge = cv2.resize(flat, None, fx=golge_scale, fy=golge_scale,
                       interpolation=cv2.INTER_CUBIC)
    golge = cv2.dilate(golge, np.ones((3, 3), np.uint8), iterations=1)

    steps.append("OCR iyileştirme uygulandı (temiz + ham + sade + ince + gölge varyantları)")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_06_enhanced.jpg"), enhanced_clean)
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_06b_enhanced_raw.jpg"), enhanced_raw)
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_06c_plain4x.jpg"), plain_4x)
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_06d_thinned.jpg"), thinned)
        cv2.imwrite(os.path.join(output_dir, f"{stem}{tag}_06e_golge.jpg"), golge)

    # ─── OCR OKUMA ───  [DEĞİŞTİ]
    # read_plate_ocr formatlama + doğrulama + skorlamayı içerir;
    # en iyi aday güven + yapısal geçerlilik + konsensüs ile seçilir
    print(f"  [7/7] OCR okuma (5 varyant × 2 strateji + oylama)...")
    plate_text, confidence, formatted, valid, score = read_plate_ocr(
        [("temiz", enhanced_clean), ("ham", enhanced_raw),
         ("sade", plain_4x), ("ince", thinned), ("gölge", golge)],
        reader
    )

    steps.append(f"OCR: '{plate_text}' → Format: '{formatted}' "
                 f"({'GEÇERLİ' if valid else 'doğrulanamadı'}, güven: {confidence:.2f})")

    return {
        'plate_text': plate_text,
        'confidence': confidence,
        'formatted': formatted,
        'valid': valid,
        'score': score,
        'steps': steps,
    }


def process_single_image(image_path, model, reader, output_dir=None, save_debug=True):
    """
    Tek bir araç fotoğrafını uçtan uca işler: tespit → temizleme → OCR → formatlama.
    
    PİPELINE AKIŞI:
    ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
    │  YOLO   │ → │ Mavi Bant │ → │ Turuncu  │ → │  Siyah   │ → │  Altlık   │
    │ Tespit  │   │  Kırpma   │   │ Sticker  │   │ Sticker  │   │ Temizleme │
    └─────────┘   └───────────┘   └──────────┘   └──────────┘   └───────────┘
                                                                       │
    ┌─────────┐   ┌───────────┐   ┌──────────────┐                     │
    │ Format  │ ← │   OCR     │ ← │ İyileştirme  │ ←──────────────────┘
    │ Düzelt  │   │   Okuma   │   │ (4x + CLAHE) │
    └─────────┘   └───────────┘   └──────────────┘
    
    Parametreler:
        image_path (str): Araç fotoğrafının dosya yolu
        model: YOLO model nesnesi
        reader: EasyOCR Reader nesnesi
        output_dir (str): Debug görüntülerin kaydedileceği klasör (None = kaydetme)
        save_debug (bool): Ara adım görüntülerini kaydet?
    
    Returns:
        result (dict): {
            'file': dosya adı,
            'plate_text': okunan metin,
            'formatted': formatlanmış metin,
            'confidence': güven oranı,
            'steps': uygulanan adımlar listesi
        }
    """
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]  # Uzantısız dosya adı
    
    print(f"\n{'='*60}")
    print(f"  İşleniyor: {filename}")
    print(f"{'='*60}")
    
    # Görüntüyü oku
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [HATA] Görüntü okunamadı: {image_path}")
        return {'file': filename, 'plate_text': None, 'formatted': '',
                'confidence': 0.0, 'valid': False, 'steps': ['HATA: Dosya okunamadı']}
    
    # ─── ADIM 1: ÇOK GEÇİŞLİ YOLO TESPİTİ + KLASİK CV YEDEĞİ ───  [DEĞİŞTİ]
    # ESKİ: tek geçiş, varsayılan imgsz=640 → yüksek çözünürlüklü
    #       fotoğraflarda plaka 640px'e küçülünce tespit edilemiyordu.
    # YENİ: 1) detect_plate_boxes() kademeli ölçek/eşiklerle dener
    #       2) YOLO tamamen başarısızsa detect_plate_classical()
    #          (blackhat morfolojisi) aday bölgeler önerir
    print(f"  [1/7] YOLO plaka tespiti (çok geçişli)...")
    gpu_available = torch.cuda.is_available()
    device = "0" if gpu_available else "cpu"

    valid_boxes, pass_no = detect_plate_boxes(model, frame, device)
    detection_source = "YOLO"

    if not valid_boxes:
        # [YENİ] Klasik CV yedek dedektörü — YOLO'nun hiç göremediği
        # plakaları morfolojik analiz ile bulur (aday kutu güveni 0
        # yazılır; asıl doğrulama OCR + yapısal kurallardadır)
        print(f"  [TESPİT] YOLO {pass_no} geçişte bulamadı → klasik CV dedektörü deneniyor...")
        classical = detect_plate_classical(frame)
        if classical:
            valid_boxes = [(cx1, cy1, cx2, cy2, 0.0, car)
                           for (cx1, cy1, cx2, cy2, _, car) in classical]
            detection_source = "klasik"
            print(f"    [KLASİK] {len(valid_boxes)} aday bölge bulundu")

    if not valid_boxes:
        print(f"  [SONUÇ] Plaka tespit EDİLEMEDİ (YOLO {pass_no} geçiş + klasik CV)!")
        return {'file': filename, 'plate_text': None, 'formatted': '',
                'confidence': 0.0, 'valid': False,
                'steps': ['Plaka tespit edilemedi (YOLO + klasik CV)']}

    # ─── ADIM 1.5: ADAY KUTU DÖNGÜSÜ ───  [YENİ: ÇOKLU PLAKA DESTEĞİ]
    # Tespit edilen aday kutuların tamamı sırayla işlenir (en fazla 5 aday)
    max_process_boxes = 5
    processed_candidates = []
    
    for idx, box in enumerate(valid_boxes[:max_process_boxes], 1):
        bx1, by1, bx2, by2, bconf, bar = box
        print(f"\n  [ADAY {idx}/{min(max_process_boxes, len(valid_boxes))}] "
              f"({bx1},{by1})-({bx2},{by2}) kaynak={detection_source} "
              f"güven={bconf:.2f} AR={bar:.1f}")

        tag = "" if idx == 1 else f"_aday{idx}"
        result = process_plate_candidate(frame, box, reader, output_dir,
                                         stem, save_debug, tag)
        if result is None:
            continue

        result['steps'].insert(0, f"{detection_source} tespit #{idx} "
                                  f"(güven: {bconf:.2f}, AR: {bar:.1f})")
        processed_candidates.append(result)

    # ─── ADIM 1.6: FİLTRELEME VE SONUÇ SENTEZİ ───
    # Kalite kapısından geçen (geçerli formatta ve makul güvenli) plakaları ayır
    good_results = [r for r in processed_candidates if r['valid'] and r['confidence'] >= 0.30]

    final_results = []
    
    if good_results:
        # En az bir geçerli plaka bulunduysa, sadece geçerli olanların tümünü döndür
        print(f"\n  [SONUÇ] {len(good_results)} geçerli plaka okundu:")
        for r in good_results:
            print(f"  ┌─────────────────────────────────────┐")
            print(f"  │  PLAKA: {r['formatted']:>12}  ({r['confidence']:.1%} güven)  │")
            print(f"  └─────────────────────────────────────┘")
            final_results.append({
                'file': filename,
                'plate_text': r['plate_text'],
                'formatted': r['formatted'],
                'confidence': r['confidence'],
                'valid': r['valid'],
                'steps': r['steps']
            })
    else:
        # Hiçbir aday geçerli plaka vermediyse, en yüksek skorlu olanı başarısız/okunamadı olarak dön
        if processed_candidates:
            best = max(processed_candidates, key=lambda x: x['score'])
            print(f"\n  [SONUÇ] Geçerli plaka formatında okuma yapılamadı (En iyi aday: {best['formatted']} %{best['confidence']:.1%})")
            final_results.append({
                'file': filename,
                'plate_text': best['plate_text'],
                'formatted': best['formatted'],
                'confidence': best['confidence'],
                'valid': best['valid'],
                'steps': best['steps']
            })
        else:
            # Hiçbir aday işlenemediyse default boş sonuç dön
            print(f"\n  [SONUÇ] Aday kutular işlenemedi!")
            final_results.append({
                'file': filename,
                'plate_text': None,
                'formatted': '',
                'confidence': 0.0,
                'valid': False,
                'steps': ['Aday kutular işlenemedi']
            })

    return final_results
    


# =============================================================================
# BÖLÜM 10: TOPLU İŞLEME VE CSV LOGLAMA
# =============================================================================

def process_all_images(folder_path, model, reader, save_debug=True):
    """
    Bir klasördeki tüm araç fotoğraflarını toplu olarak işler.
    
    Desteklenen formatlar: .jpg, .jpeg, .png, .bmp, .tiff
    
    Parametreler:
        folder_path (str): Araç fotoğraflarının bulunduğu klasör yolu
        model: YOLO model nesnesi
        reader: EasyOCR Reader nesnesi
        save_debug (bool): Ara adım görüntülerini kaydet?
    
    Returns:
        all_results (list[dict]): Her görüntü için sonuç sözlükleri
    """
    # Desteklenen görüntü uzantıları
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"[HATA] '{folder_path}' klasöründe görüntü bulunamadı!")
        return []
    
    # Sıralı işleme (tutarlı çıktı için)
    image_files.sort()
    
    print(f"\n{'#'*60}")
    print(f"  TOPLU PLAKA TESPİT & TEMİZLEME")
    print(f"  Klasör: {folder_path}")
    print(f"  Görüntü sayısı: {len(image_files)}")
    print(f"{'#'*60}")
    
    # Debug çıktı dizini - Her çalıştırmada temiz başlangıç için override edilir
    output_dir = os.path.join(folder_path, "sonuclar")
    if save_debug:
        import shutil
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"[TEMİZLİK] Eski sonuçlar klasörü '{output_dir}' silindi.")
            except Exception as e:
                print(f"[UYARI] Eski sonuçlar klasörü temizlenemedi: {e}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    all_results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] ", end="")
        results = process_single_image(image_path, model, reader, output_dir, save_debug)
        all_results.extend(results)
    
    return all_results


def log_results_to_csv(results, csv_path="plaka_log.csv", overwrite=True):
    """
    İşlem sonuçlarını temiz ve okunaklı CSV dosyasına kaydeder.
    
    Sütun yapısı:
    ┌────────────┬─────────────────────┬────────┐
    │ Plaka      │ Tarih_Saat          │ Guven% │
    ├────────────┼─────────────────────┼────────┤
    │ 34 N 5953  │ 2026-07-07 12:00:00 │  87.4  │
    └────────────┴─────────────────────┴────────┘
    
    Tespit edilemeyen plakalar satıra yazılmaz, atlanır.
    
    Parametreler:
        results (list[dict]): process_all_images'tan dönen sonuçlar
        csv_path (str): CSV dosya yolu
        overwrite (bool): True → dosyayı her seferinde sıfırdan yaz
                          False → mevcut dosyaya ekle (append modu)
    """
    write_mode = 'w' if overwrite else 'a'
    file_exists = os.path.exists(csv_path) and not overwrite
    written_count = 0  # [YENİ] Gerçekten yazılan satır sayısı (filtre sonrası)

    with open(csv_path, mode=write_mode, newline='', encoding='utf-8-sig') as f:
        # utf-8-sig: Excel'in Türkçe karakterleri doğru göstermesi için BOM ekler
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Plaka", "Tarih_Saat", "Guven_%"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in results:
            formatted  = result.get('formatted', '')
            confidence = result.get('confidence', 0.0)
            valid      = result.get('valid', False)

            # [DEĞİŞTİ] KALİTE KAPISI: sadece güvenilir okumaları logla.
            # 1. Yapısal olarak geçersiz okumalar çöp veridir ('06 J 003')
            # 2. Geçerli görünse bile güveni %35'in altındaki okumalar
            #    güvenilmezdir (tek zayıf stratejiden gelen yarı-doğru
            #    okumalar: '06 ABL 328' gibi)
            # CSV'ye yanlış kayıt yazmak hiç yazmamaktan kötüdür.
            if not formatted or confidence <= 0:
                continue
            if not valid or confidence < 0.35:
                continue
            
            writer.writerow([
                formatted,                        # Plaka (örn. "34 N 5953")
                timestamp,                        # Tarih ve saat
                f"{confidence * 100:.1f}",        # Güven % (örn. "87.4")
            ])
            written_count += 1

    # [DEĞİŞTİ] Gerçek yazılan satır sayısını raporla (eski mesaj filtre
    # edilenleri de sayıyordu — yanıltıcıydı)
    skipped = len(results) - written_count
    print(f"\n[CSV] {written_count} güvenilir okuma '{csv_path}' dosyasına yazıldı"
          f"{f' ({skipped} düşük kaliteli sonuç filtrelendi)' if skipped else ''}.")



def print_summary_table(results):
    """
    Sonuçları konsola güzel formatlanmış tablo olarak yazdırır.
    
    Parametreler:
        results (list[dict]): İşlem sonuçları
    """
    print(f"\n{'='*80}")
    print(f"{'PLAKA TESPİT SONUÇLARI':^80}")
    print(f"{'='*80}")
    print(f"{'Dosya':<45} {'Plaka':<15} {'Güven':>8} {'Format':>8}")
    print(f"{'-'*45} {'-'*15} {'-'*8} {'-'*8}")

    success_count = 0
    suspect_count = 0
    for r in results:
        formatted = r.get('formatted', '-')
        confidence = r.get('confidence', 0.0)
        valid = r.get('valid', False)
        # [DEĞİŞTİ] Üç durumlu değerlendirme:
        # ✓ = yapısal geçerli + yeterli güven (CSV'ye de yazılır)
        # ? = yapısal geçerli ama düşük güven (şüpheli — CSV'ye yazılmaz)
        # ✗ = geçersiz/okunamadı
        if formatted and formatted != '-' and valid and confidence >= 0.35:
            status = "✓"
            success_count += 1
        elif formatted and formatted != '-' and valid:
            status = "?"
            suspect_count += 1
        else:
            status = "✗"

        valid_str = "GEÇERLİ" if valid else "-"
        print(f"{status} {r['file']:<43} {formatted:<15} {confidence:>7.1%} {valid_str:>8}")

    print(f"{'-'*80}")
    print(f"  Toplam: {len(results)} görüntü | "
          f"Başarılı: {success_count} | "
          f"Şüpheli: {suspect_count} | "
          f"Başarısız: {len(results) - success_count - suspect_count}")
    print(f"{'='*80}")


# =============================================================================
# ANA ÇALIŞTIRMA BLOĞU
# =============================================================================

if __name__ == "__main__":
    # Model ve OCR motorunu başlat
    model = initialize_model()
    if model is None:
        print("Model yüklenemedi. Çıkılıyor...")
        exit(1)
    
    reader, gpu_available = initialize_ocr()
    
    # ─── TOPLU İŞLEME: arabalar/ klasöründeki tüm görselleri işle ───
    IMAGES_FOLDER = "arabalar"
    
    if os.path.isdir(IMAGES_FOLDER):
        # Tüm görselleri işle
        results = process_all_images(IMAGES_FOLDER, model, reader, save_debug=True)
        
        # Sonuç tablosu yazdır
        print_summary_table(results)
        
        # CSV'ye kaydet
        log_results_to_csv(results)
    else:
        # Klasör yoksa tek görüntü dene
        print(f"[UYARI] '{IMAGES_FOLDER}' klasörü bulunamadı.")
        fallback_images = ["araba1.jpg", "PlateDetection.png"]
        for img in fallback_images:
            if os.path.exists(img):
                print(f"Tek görüntü işleniyor: {img}")
                results = process_single_image(img, model, reader, "sonuclar", True)
                print_summary_table(results)
                log_results_to_csv(results)
                break
        else:
            print("[HATA] İşlenecek görüntü bulunamadı!")
