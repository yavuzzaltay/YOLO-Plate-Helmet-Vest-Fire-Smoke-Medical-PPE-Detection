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
    YOLO modelini yükler. Önce mevcut dizinde 'best.pt' arar,
    bulamazsa runs/ altındaki en son eğitim çıktısını kullanır.
    
    Returns:
        model: YOLO model nesnesi veya None (bulunamazsa)
    """
    if os.path.exists("best.pt"):
        best_weight = "best.pt"
    else:
        # Eğitim çıktılarını tara — en son tarihli olanı al
        weight_files = glob.glob("runs/detect/train*/weights/best.pt")
        best_weight = sorted(weight_files)[-1] if weight_files else None

    if best_weight:
        print(f"[MODEL] Ağırlık dosyası yüklendi: {best_weight}")
        return YOLO(best_weight)
    else:
        print("[HATA] 'best.pt' bulunamadı. Eğitim tamamlanmamış olabilir.")
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
    # ─── 1. BÜYÜTME (4x) ───
    # INTER_CUBIC: bikübik interpolasyon
    # - 4x4 piksel komşuluk kullanır (INTER_LINEAR sadece 2x2 kullanır)
    # - Daha yavaş ama daha kaliteli büyütme
    # - Kenarlar daha pürüzsüz, pikselleşme az
    # fx=4, fy=4: her iki eksende 4 kat büyüt
    plate_large = cv2.resize(plate_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
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

def read_plate_ocr(plate_image, reader):
    """
    Temizlenmiş ve iyileştirilmiş plaka görüntüsünden metin okur.
    Birden fazla OCR stratejisi dener ve en iyi sonucu seçer.
    
    ÇOK STRATEJİLİ YAKLAŞIM — Neden?
    Tek bir OCR çağrısı her zaman en iyi sonucu vermez çünkü:
    - Renkli görüntüde iyi okunan bazı karakterler binarize'da kaybolabilir
    - Binarize'da netleşen bazı karakterler renkli'de gürültüye karışabilir
    - Bu yüzden hem renkli hem de binarize versiyonu deneriz
    
    ALLOWLIST MANTIĞI:
    Türk plakalarında kullanılan karakterler sınırlıdır:
    - Rakamlar: 0-9
    - Harfler: A B C D E F G H I J K L M N O P R S T U V Y Z
    - Türk plakalarında Q, W, X harfleri KULLANILMAZ
    - allowlist ile OCR'ı bu karakterlere kısıtlarız → doğruluk artar
    - Örn: 'Q' yerine 'O', 'W' yerine 'V' okunma hatası önlenir
    
    Parametreler:
        plate_image (np.ndarray): İyileştirilmiş plaka görüntüsü
        reader: EasyOCR Reader nesnesi
    
    Returns:
        best_text (str): Okunan plaka metni (veya None)
        best_confidence (float): Güven oranı (0.0-1.0)
    """
    # Türk plakalarında kullanılabilecek tüm karakterler
    PLATE_ALLOWLIST = 'ABCDEFGHIJKLMNOPRSTUVYZ0123456789'
    
    candidates = []
    
    # ─── STRATEJİ 1: Renkli (iyileştirilmiş) görüntüde OCR ───
    try:
        result_color = reader.readtext(
            plate_image,
            allowlist=PLATE_ALLOWLIST,
            paragraph=False  # Her metin bölgesini ayrı ayrı oku
        )
        if result_color:
            # bbox'ları soldan sağa sırala (plaka sola-sağa okunur)
            result_color.sort(key=lambda x: x[0][0][0])
            text = "".join([t for (_, t, _) in result_color])
            conf = sum([c for (_, _, c) in result_color]) / len(result_color)
            candidates.append((text, conf, "renkli"))
    except Exception as e:
        print(f"    [OCR] Renkli okuma hatası: {e}")
    
    # ─── STRATEJİ 2: Binarize (siyah-beyaz) görüntüde OCR ───
    try:
        binary = adaptive_binarize(plate_image)
        result_binary = reader.readtext(
            binary,
            allowlist=PLATE_ALLOWLIST,
            paragraph=False
        )
        if result_binary:
            result_binary.sort(key=lambda x: x[0][0][0])
            text = "".join([t for (_, t, _) in result_binary])
            conf = sum([c for (_, _, c) in result_binary]) / len(result_binary)
            candidates.append((text, conf, "binarize"))
    except Exception as e:
        print(f"    [OCR] Binarize okuma hatası: {e}")
    
    # ─── STRATEJİ 3: Gri tonlama + OTSU eşikleme ile OCR ───
    try:
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result_otsu = reader.readtext(
            otsu,
            allowlist=PLATE_ALLOWLIST,
            paragraph=False
        )
        if result_otsu:
            result_otsu.sort(key=lambda x: x[0][0][0])
            text = "".join([t for (_, t, _) in result_otsu])
            conf = sum([c for (_, _, c) in result_otsu]) / len(result_otsu)
            candidates.append((text, conf, "otsu"))
    except Exception as e:
        print(f"    [OCR] OTSU okuma hatası: {e}")
    
    if not candidates:
        print("    [OCR] Hiçbir strateji sonuç vermedi!")
        return None, 0.0
    
    # En iyi sonucu seç: en yüksek güven oranı
    best = max(candidates, key=lambda x: x[1])
    print(f"    [OCR] Sonuçlar:")
    for text, conf, strategy in candidates:
        marker = " ✓" if (text, conf) == (best[0], best[1]) else ""
        print(f"           {strategy:>8}: '{text}' (güven: {conf:.2f}){marker}")
    
    return best[0], best[1]


# =============================================================================
# BÖLÜM 8: TÜRK PLAKA FORMAT DÜZELTMESİ
# =============================================================================

def format_turkish_plate(raw_text):
    """
    OCR çıktısını standart Türk plaka formatına düzeltir.
    
    TÜRK PLAKA FORMATI:
    ┌─────────────────────────────────┐
    │  İL KODU  HARF SERİSİ  NUMARA  │
    │    XX       YYY          ZZZZ   │
    └─────────────────────────────────┘
    
    - İl kodu: 01-81 arası 2 haneli rakam
    - Harf serisi: 1-3 harf (A-Z, Q/W/X hariç)
    - Numara: 2-4 haneli rakam
    
    Geçerli format örnekleri:
    34 ABC 1234, 06 A 0001, 42 AEH 738
    
    REGEX AÇIKLAMASI:
    ^(\\d{2})([A-Z]{1,3})(\\d{2,4})$
    ^           → Metnin başı
    (\\d{2})    → Tam 2 rakam (il kodu) — Grup 1
    ([A-Z]{1,3}) → 1 ila 3 büyük harf (seri) — Grup 2
    (\\d{2,4})  → 2 ila 4 rakam (numara) — Grup 3
    $           → Metnin sonu
    
    EK DÜZELTMELER:
    - Yaygın OCR hataları düzeltilir (0↔O, 1↔I, 8↔B gibi)
    - Pozisyon bazlı: il kodu bölgesinde harf varsa rakama çevir,
      harf bölgesinde rakam varsa harfe çevir
    
    Parametreler:
        raw_text (str): OCR'dan gelen ham metin
    
    Returns:
        formatted (str): Formatlanmış plaka metni (XX YYY ZZZZ)
    """
    if not raw_text:
        return ""
    
    # Temizle: boşluk, tire, nokta kaldır + büyük harfe çevir
    text = raw_text.upper().replace(" ", "").replace("-", "").replace(".", "")
    
    # ─── YAYGN OCR HATA DÜZELTMELERİ ───
    
    # Önce direkt regex dene (temiz okuma durumu)
    match = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    
    # Eşleşmezse pozisyon bazlı düzeltme yap
    # İl kodu bölgesi (ilk 2 karakter): rakam olmalı
    # Harf bölgesi (ortadaki 1-3 karakter): harf olmalı
    # Numara bölgesi (son 2-4 karakter): rakam olmalı
    
    # Harf → Rakam dönüşüm tablosu (OCR'ın karıştırdığı benzer şekiller)
    letter_to_digit = {
        'O': '0', 'Q': '0',  # O ve Q → 0
        'I': '1', 'L': '1',  # I ve L → 1
        'Z': '2',             # Z → 2
        'S': '5',             # S → 5
        'G': '6',             # G → 6
        'T': '7',             # T → 7
        'B': '8',             # B → 8
    }
    
    # Rakam → Harf dönüşüm tablosu
    digit_to_letter = {
        '0': 'O',  # 0 → O
        '1': 'I',  # 1 → I
        '2': 'Z',  # 2 → Z
        '5': 'S',  # 5 → S
        '6': 'G',  # 6 → G
        '8': 'B',  # 8 → B
    }
    
    # Minimum 5 karakter (XX Y ZZ) olmalı
    if len(text) < 5:
        return text
    
    # İlk 2 karakter → rakam yapma denemesi
    corrected = list(text)
    for i in range(min(2, len(corrected))):
        if corrected[i].isalpha() and corrected[i] in letter_to_digit:
            corrected[i] = letter_to_digit[corrected[i]]
    
    # Son kısım → rakam yapma denemesi (sondan 2-4 karakter)
    # Ortadaki harfleri bul: ilk 2 rakamdan sonra, son rakamlardan önce
    text_corrected = "".join(corrected)
    match = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', text_corrected)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    
    # Hâlâ eşleşmiyorsa, harf bölgesindeki rakamları harfe çevirmeyi dene
    if len(text) >= 5:
        part1 = text[:2]   # İl kodu
        remaining = text[2:]
        
        # Harfleri ve rakamları ayır
        letters = ""
        for ch in remaining:
            if ch.isalpha():
                letters += ch
            elif ch in digit_to_letter and len(letters) < 3 and not any(c.isdigit() for c in letters):
                letters += digit_to_letter[ch]
            else:
                break
        
        numbers = remaining[len(letters):] if letters else remaining[1:]
        
        # İl kodunu düzelt
        fixed_il = ""
        for ch in part1:
            if ch.isdigit():
                fixed_il += ch
            elif ch in letter_to_digit:
                fixed_il += letter_to_digit[ch]
            else:
                fixed_il += ch
        
        # Numara kısmını düzelt
        fixed_num = ""
        for ch in numbers:
            if ch.isdigit():
                fixed_num += ch
            elif ch in letter_to_digit:
                fixed_num += letter_to_digit[ch]
            else:
                fixed_num += ch
        
        if len(fixed_il) == 2 and 1 <= len(letters) <= 3 and 2 <= len(fixed_num) <= 4:
            return f"{fixed_il} {letters} {fixed_num}"
    
    # Hiçbir düzeltme çalışmadıysa ham metni döndür
    return text


# =============================================================================
# BÖLÜM 9: TEK GÖRÜNTÜ İŞLEME PİPELINE'I
# =============================================================================

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
                'confidence': 0.0, 'steps': ['HATA: Dosya okunamadı']}
    
    # ─── ADIM 1: YOLO ile plaka tespiti ───
    print(f"  [1/7] YOLO plaka tespiti...")
    gpu_available = torch.cuda.is_available()
    device = "0" if gpu_available else "cpu"
    # Düşük conf eşiği (0.25) kullanıyoruz çünkü:
    # - Bazı açılardan çekilen plakalar düşük güvenle tespit edilebilir
    # - Aspect ratio filtresi ile yanlış tespitleri zaten eleyeceğiz
    results = model.predict(source=image_path, conf=0.25, device=device, verbose=False)
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"  [SONUÇ] Plaka tespit EDİLEMEDİ!")
        return {'file': filename, 'plate_text': None, 'formatted': '', 
                'confidence': 0.0, 'steps': ['Plaka tespit edilemedi']}
    
    # ─── PLAKA DOĞRULAMA (Aspect Ratio Filtresi) ───
    # YOLO bazen plaka olmayan nesneleri de tespit edebilir:
    # - Watermark yazıları (1779045735949 fotoğrafındaki #1313759809)
    # - Araba logoları, far kenarları vb.
    #
    # Türk plakası fiziksel ölçüleri: 520mm x 110mm → aspect ratio ≈ 4.7:1
    # Kabul edilebilir aralık: 2.0 - 7.0 (açı ve perspektif farkları için geniş)
    # - 2.0: çok eğik açıdan görülen plakalar
    # - 7.0: uzun ama dar perspektif
    #
    # Minimum boyut: en az 30px genişlik ve 10px yükseklik
    # (bundan küçük tespitler OCR için kullanılamaz)
    
    PLATE_AR_MIN = 1.5   # Minimum aspect ratio (düşük açı ve kırpma toleransı)
    PLATE_AR_MAX = 7.0   # Maximum aspect ratio
    MIN_WIDTH = 30        # Minimum plaka genişliği (piksel)
    MIN_HEIGHT = 10       # Minimum plaka yüksekliği (piksel)
    
    # Tüm tespitleri aspect ratio ile filtrele, güvene göre sırala
    valid_boxes = []
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i].xyxy[0])
        bw = bx2 - bx1
        bh = by2 - by1
        bconf = float(boxes[i].conf[0])
        
        if bh <= 0 or bw <= 0:
            continue
            
        ar = bw / bh  # Aspect ratio
        
        if (PLATE_AR_MIN <= ar <= PLATE_AR_MAX and 
            bw >= MIN_WIDTH and bh >= MIN_HEIGHT):
            valid_boxes.append((bx1, by1, bx2, by2, bconf, ar))
            print(f"    Tespit #{i+1}: ({bx1},{by1})-({bx2},{by2}) "
                  f"AR={ar:.1f} Güven={bconf:.2f} ✓ GEÇERLİ")
        else:
            print(f"    Tespit #{i+1}: ({bx1},{by1})-({bx2},{by2}) "
                  f"AR={bw/bh:.1f} Güven={bconf:.2f} ✗ REDDEDİLDİ "
                  f"({'AR dışı' if not (PLATE_AR_MIN <= ar <= PLATE_AR_MAX) else 'çok küçük'})")
    
    if not valid_boxes:
        print(f"  [SONUÇ] Geçerli plaka tespiti bulunamadı (aspect ratio filtresi)")
        return {'file': filename, 'plate_text': None, 'formatted': '', 
                'confidence': 0.0, 'steps': ['Tespit var ama plaka boyutlarına uymuyor']}
    
    # En yüksek güvenli geçerli kutuyu seç
    valid_boxes.sort(key=lambda x: x[4], reverse=True)
    x1, y1, x2, y2, detect_conf, ar = valid_boxes[0]
    
    plate_crop = frame[y1:y2, x1:x2]
    print(f"  [1/7] Plaka bulundu! Konum: ({x1},{y1})-({x2},{y2}), "
          f"Güven: {detect_conf:.2f}, AR: {ar:.1f}")
    
    steps = [f"YOLO tespit (güven: {detect_conf:.2f}, AR: {ar:.1f})"]
    
    # Debug kayıt dizini oluştur
    if save_debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_01_original_crop.jpg"), plate_crop)
    
    current = plate_crop.copy()
    
    # ─── ADIM 2: Mavi TR bandı ───
    print(f"  [2/7] Mavi TR bandı kontrolü...")
    current, blue_found, blue_width = detect_blue_band(current)
    if blue_found:
        steps.append(f"Mavi bant kırpıldı ({blue_width}px)")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_02_no_blue.jpg"), current)
    
    # ─── ADIM 3: Turuncu sticker ───
    print(f"  [3/7] Turuncu sticker kontrolü...")
    current, orange_found = detect_and_remove_orange_sticker(current)
    if orange_found:
        steps.append("Turuncu sticker temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_03_no_orange.jpg"), current)
    
    # ─── ADIM 4: Siyah sticker ───
    print(f"  [4/7] Siyah sticker kontrolü...")
    current, black_found = detect_and_remove_black_sticker(current)
    if black_found:
        steps.append("Siyah sticker temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_04_no_black.jpg"), current)
    
    # ─── ADIM 5: Plaka altlığı ve çerçeve ───
    print(f"  [5/7] Plaka altlığı/çerçeve kontrolü...")
    current = remove_plate_frame_and_holder(current)
    steps.append("Altlık/çerçeve temizlendi")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_05_no_frame.jpg"), current)
    
    # ─── ADIM 6: OCR için iyileştirme ───
    print(f"  [6/7] Görüntü iyileştirme (4x büyütme + CLAHE + keskinleştirme)...")
    enhanced = enhance_plate_for_ocr(current)
    steps.append("OCR iyileştirme uygulandı")
    if save_debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_06_enhanced.jpg"), enhanced)
        # OCR'ın beslendiği siyah-beyaz binarize halini de kaydedelim
        binarized = adaptive_binarize(enhanced)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_07_binarized.jpg"), binarized)
    
    # ─── ADIM 7: OCR okuma ───
    print(f"  [7/7] OCR okuma (çok stratejili)...")
    plate_text, confidence = read_plate_ocr(enhanced, reader)
    
    # Format düzelt
    formatted = format_turkish_plate(plate_text)
    steps.append(f"OCR: '{plate_text}' → Format: '{formatted}' (güven: {confidence:.2f})")
    
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  SONUÇ: {formatted:>12}  ({confidence:.1%} güven)  │")
    print(f"  └─────────────────────────────────────┘")
    
    return {
        'file': filename,
        'plate_text': plate_text,
        'formatted': formatted,
        'confidence': confidence,
        'steps': steps
    }


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
        result = process_single_image(image_path, model, reader, output_dir, save_debug)
        all_results.append(result)
    
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
    
    with open(csv_path, mode=write_mode, newline='', encoding='utf-8-sig') as f:
        # utf-8-sig: Excel'in Türkçe karakterleri doğru göstermesi için BOM ekler
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Plaka", "Tarih_Saat", "Guven_%"])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for result in results:
            formatted  = result.get('formatted', '')
            confidence = result.get('confidence', 0.0)
            
            # Tespit edilemeyen plakaları atla — sadece okunan plakaları yaz
            if not formatted or confidence <= 0:
                continue
            
            writer.writerow([
                formatted,                        # Plaka (örn. "34 N 5953")
                timestamp,                        # Tarih ve saat
                f"{confidence * 100:.1f}",        # Güven % (örn. "87.4")
            ])
    
    print(f"\n[CSV] {len(results)} sonuç '{csv_path}' dosyasına yazıldı.")



def print_summary_table(results):
    """
    Sonuçları konsola güzel formatlanmış tablo olarak yazdırır.
    
    Parametreler:
        results (list[dict]): İşlem sonuçları
    """
    print(f"\n{'='*80}")
    print(f"{'PLAKA TESPİT SONUÇLARI':^80}")
    print(f"{'='*80}")
    print(f"{'Dosya':<45} {'Plaka':<15} {'Güven':>8}")
    print(f"{'-'*45} {'-'*15} {'-'*8}")
    
    success_count = 0
    for r in results:
        formatted = r.get('formatted', '-')
        confidence = r.get('confidence', 0.0)
        status = "✓" if formatted and formatted != '-' and confidence > 0.3 else "✗"
        
        if status == "✓":
            success_count += 1
        
        print(f"{status} {r['file']:<43} {formatted:<15} {confidence:>7.1%}")
    
    print(f"{'-'*80}")
    print(f"  Toplam: {len(results)} görüntü | "
          f"Başarılı: {success_count} | "
          f"Başarısız: {len(results) - success_count}")
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
                result = process_single_image(img, model, reader, "sonuclar", True)
                print_summary_table([result])
                log_results_to_csv([result])
                break
        else:
            print("[HATA] İşlenecek görüntü bulunamadı!")
