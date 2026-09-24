# Plaka video işleme mantığı — sıfırdan, detaylı anlatım

Bu dosya sadece şunu anlatıyor: **video/canlı kamerada plaka tespiti nasıl çalışıyor** (`PlateDetection/video_plaka.py` + `app.py` içindeki `run_plate_video`). Baret/yelek tespiti bu karmaşıklığın hiçbirine ihtiyaç duymuyor — o, tek karede YOLO'nun verdiği cevabı (baret var/yok) doğrudan kullanıyor, çünkü orada "okuma" diye bir şey yok, sınıflandırma yeterli. Plaka farklı: YOLO sadece "şurada bir plaka var" der, asıl zor iş o kutunun içindeki yazıyı **doğru okumak** (OCR), ve tek kareden okumak güvenilmez olduğu için bütün bu sistem kuruldu.

---

## 1. Neden bu kadar karmaşık? (tek cümlede özet)

Tek bir karede plaka gölgeli, bulanık, açılı ya da parlamalı görünebilir → OCR yanlış okur. Ama video/kamera aynı plakayı **onlarca farklı anda, farklı açı/ışıkla** gösterir. Sistem bunu şöyle kullanıyor: aynı plakayı birden çok bağımsız anda yakala, her birini ayrı ayrı oku, sonuçları karşılaştır (oyla), ancak yeterince kanıt birikince CSV'ye yaz. Amaç: **CSV'ye yanlış plaka yazmak, hiç yazmamaktan daha kötü.**

---

## 2. Temel kavramlar sözlüğü

| Terim | Ne demek | Örnek |
|---|---|---|
| **Kare (frame)** | Videonun tek bir fotoğrafı | 60 fps video = saniyede 60 kare |
| **Örnek (sample)** | İşlemeye değer bulup seçtiğimiz kare; geri kalanı hiç işlenmez | dosyada her 6. kare, canlıda 0.25 sn'de 1 kare |
| **Kutu (box)** | YOLO'nun "burada plaka var" dediği dikdörtgen koordinatı | `(x1,y1,x2,y2)` |
| **İz / Track** | Aynı fiziksel plakanın zaman içindeki takibi. İz = Track, aynı şey, iki dilde adı | ekrandaki `iz#3` etiketi |
| **Kırpım (crop)** | Örnek kareden, plaka kutusunun etrafı kesilerek çıkarılan küçük fotoğraf | izin "delil koleksiyonu"ndaki her bir parça |
| **IoU** | İki kutunun ne kadar örtüştüğü (kesişim/birleşim oranı, 0-1 arası) | 1.0 = birebir aynı yer, 0 = hiç örtüşmüyor |
| **OCR** | Kırpımdaki yazıyı metne çeviren motor (EasyOCR) | en pahalı, en yavaş adım |
| **Çapraz doğrulama** | Sonucun CSV'ye yazılabilmesi için gereken kanıt şartı | ≥2 kırpım aynı sonucu verdi VEYA tek okuma güveni ≥ %80 |
| **Çözülmüş iz (resolved)** | Çapraz doğrulama sağlanmış, artık OCR'a gerek kalmamış iz | ekranda camgöbeği kutu + üstünde plaka yazısı |

---

## 3. Bütün sabitler — ne işe yarıyor, neden bu değer

Bunların hepsi `video_plaka.py` dosyasının başında tanımlı:

| Sabit | Değer | Ne yapar | Neden bu değer |
|---|---|---|---|
| `FRAME_SKIP` | 6 | Dosya modunda her kaçıncı karenin işleneceği | 60fps'te saniyede ~10 örnek — plaka en az ~1 sn ekranda kalır, kaçmaz |
| `YOLO_CONF` | 0.15 | YOLO'nun kutu kabul eşiği (düşük!) | Video bağlamında düşük eşik güvenli — yanlış kutuları zaten iz takibi ve OCR eleyecek |
| `IOU_MATCH` | 0.30 | Yeni kutu, var olan izle "aynı plaka" sayılsın diye gereken minimum örtüşme | Deneyle bulunmuş dengeli eşik — düşükse farklı plakalar birleşir, yüksekse aynı plaka bölünür |
| `MISS_LIMIT` | 15 örnek | Bir iz kaç örnek boyunca hiç güncellenmezse kapanır | ~1.5 sn (10Hz örneklemeye göre hesaplanmış — moda göre gerçek karşılığı değişir, bkz. §6) |
| `BEST_CROPS` | 5 | Bir OCR denemesinde en fazla kaç kırpımın tam işleneceği | Deneyde tek kırpımlık eksik okuma yanlış çıktı — daha çok bağımsız kırpım = daha sağlam oylama |
| `KEEP_CROPS` | 8 | İz başına bellekte tutulan en iyi kırpım sayısı | Bellek sınırı — sınırsız biriktirmek gereksiz, kalite sıralamasıyla en iyi 8 yeter |
| `CROP_GAP` | 0.4 sn | Saklanan kırpımlar arası minimum zaman farkı | Ardışık kareler neredeyse özdeş — zamana yayılmamış kırpımlar bağımsız kanıt SAYILMAZ |
| `CROP_PAD` | 0.45 | Kırpım kenar payı (plaka yüksekliğinin oranı) | Eğiklik düzeltme için plaka sınırının biraz fazlası görünmeli |
| `MIN_BOX_W` | 60 px | Bundan dar kutu hiç kırpım olarak saklanmaz | Bu kadar dar bir plaka OCR için zaten umutsuz |
| `DEDUP_WINDOW` | 120 sn | Aynı plaka bu süre içinde tekrar loglanmaz | Dashcam'de öndeki araç dakikalarca takip edilebilir — her görünüş yeni kayıt değildir |
| `SAMPLE_PERIOD` (app.py, canlı mod) | 0.25 sn | Canlı modda örnekleme sıklığı | `FRAME_SKIP`in dosya modundaki oranıyla aynı mertebe (~4 Hz) |

---

## 4. Baştan sona akış — her adım gerçek kodla

### Adım 0 — Video/kamera açılışı
```python
cap = cv2.VideoCapture(video_path)   # dosya yolu VEYA "http://telefon:8080/video" URL'si
```
OpenCV için dosya ile canlı URL arasında fark yok, ikisini de "kare kaynağı" olarak açar.

### Adım 1 — Örnekleme (hangi kareler işlenecek)

**Dosya modu** — kare **indeksine** bakar:
```python
frame_no += 1
if frame_no % FRAME_SKIP != 0:   # 6'da 1 değilse
    continue                     # atla, işleme
video_sec = frame_no / fps       # kare no'yu saniyeye çevir
```

**Canlı mod** — duvar saatine bakar:
```python
video_sec = time.time() - t0
if video_sec - last_sample < SAMPLE_PERIOD:   # 0.25 sn dolmadıysa
    continue
last_sample = video_sec
```

Neden farklı? **Dosya modu tekrarlanabilirlik ister** (aynı dosyayı iki kez işlesen, PC hızlı da olsa yavaş da olsa, hep aynı kareler seçilsin). **Canlı mod güncellik ister** (kaynağı biz kontrol etmiyoruz, işleme yavaş kalırsa geriye düşmemek için zamana göre karar vermeliyiz). Detaylı gerekçe için §6.

Bu adımda ayrıca `cap.grab()` (kareyi ağdan/dosyadan çeker, decode ETMEZ — ucuz) ile `cap.retrieve()` (decode eder — pahalı) ayrımı kullanılır: işlemeyeceğimiz kareler hiç decode edilmeden atılır.

### Adım 2 — YOLO tespiti (her seçilen örnekte çalışır)
```python
detections = detect_boxes_fast(model, frame, device, imgsz)
# → [(x1,y1,x2,y2,güven), ...]
```
Tek geçişli, hızlı bir YOLO çağrısı (fotoğraf modundaki 3 geçişliden farklı — video zaten zamansal fazlalık sağlıyor, bir karede kaçan plaka sonraki örnekte yakalanır). Sonuç: bu karede bulunan tüm plaka **kutuları**.

### Adım 3 — İz eşleştirme (her bulunan kutu için)
```python
for det in detections:
    box = det[:4]
    best_track, best_iou = None, IOU_MATCH
    for tr in active_tracks:
        iou = _iou(box, tr.box)
        if iou > best_iou:
            best_track, best_iou = tr, iou
    if best_track is None:
        best_track = PlateTrack(box, sample_idx, video_sec)   # YENİ iz
    else:
        best_track.update(box, sample_idx, video_sec)          # VAR OLAN izi güncelle
```
Mantık: "Bu kutu, hâlâ açık olan izlerden hangisiyle en çok örtüşüyor (IoU > 0.30)? Biriyle örtüşüyorsa o fiziksel plakanın devamı — izi güncelle. Hiçbiriyle örtüşmüyorsa yeni bir plaka gördük — yeni iz aç."

### Adım 4 — Kırpım biriktirme (her eşleşen kutu için)
```python
best_track.add_crop(frame, box, det[4], video_sec)
```
İçeride:
1. Kutu çok darsa (`< MIN_BOX_W`) hiç saklanmaz.
2. Kutunun etrafına pay eklenip (`CROP_PAD`) alt-kare kesilir.
3. Kalite hesaplanır: `genişlik × netlik` (netlik = Laplace varyansı — bulanık görüntüde düşer).
4. **Zamansal çeşitlilik kontrolü:** zaten saklanan bir kırpımdan `CROP_GAP=0.4` sn'den az uzaktaysa, ikisinden kaliteli olan kalır (yeni slot açılmaz).
5. Yeterince uzaksa yeni kırpım olarak eklenir, kalite sırasına göre sıralanır, `KEEP_CROPS=8`'i aşan en kötüler silinir.

**Neden bu kadar titiz?** Çünkü OCR oylaması ancak kırpımlar **birbirinden bağımsız** olursa işe yarar. Art arda 3 kare pratikte aynı fotoğraftır — üçü de aynı hatayı yapar, oylama hiçbir şey kazandırmaz. 0.4 saniye arayla alınan kırpımlar farklı ışık/titreşim/açı taşır — gerçek bağımsız kanıt budur.

### Adım 5 — OCR ne zaman tetiklenir (artımlı kontrol)
```python
if not best_track.resolved:
    n = len(best_track.crops)
    first_try = best_track.last_attempt_count == 0 and n >= 3
    if first_try or n - best_track.last_attempt_count >= 2:
        result = attempt_read(best_track, reader, ...)
```
Kural: **ilk deneme kırpım sayısı 3'e ulaşınca**, **sonraki denemeler her +2 yeni kırpımda bir**. Neden 3? 1-2 örneklik izler neredeyse hep yanlış tespit (tabela, reklam yazısı) — duman testinde bunların OCR sonuçları hep çöp çıkmış. Neden "her +2"? Sürekli her kırpımda tekrar denemek (pahalı OCR'ı) israf olurdu; 2 yeni kanıt birikince tekrar bakmaya değer.

**Bu, video hâlâ oynarken** (iz hâlâ açıkken) çalışan bir kontrol — "izin kapanmasını bekleme, elindeki kanıt yeterliyse hemen karar ver" mantığı. Eskiden sadece iz kapanınca (araç kadrajdan çıkınca) OCR çalışıyordu; bu, tam önünüzde uzun süre aynı hızda giden bir aracın plakasını sonsuza dek işlemeden bırakıyordu.

### Adım 6 — `attempt_read` içinde ne oluyor (tek bir deneme)
1. Kırpımlardan **en iyi `BEST_CROPS=5` tanesi** alınır.
2. Her biri tam OCR hattından geçirilir (`process_plate_candidate`) — bu da kendi içinde her kırpım için 5 görüntü varyantı (temiz/ham/sade/ince/gölge) × birkaç strateji dener. Yani **1 kırpım ≈ 10 EasyOCR çağrısı**.
3. Geçerli (Türk plaka formatına uyan) okumalar arasında **karakter bazlı oylama** yapılır (`_character_vote`): aynı uzunluk/yapıdaki adaylar pozisyon pozisyon karşılaştırılır, her pozisyonda en çok güven toplayan karakter kazanır.
4. **Kareler arası konsensüs:** aynı sonucu veren bağımsız kırpım sayısına göre güven yükseltilir — ≥3 kırpım aynıysa güven tabanı %60'a çıkar, =2 kırpım aynıysa sadece gerçek güvenlerin en yükseği alınır (taban yok, çünkü 2 kırpım sistematik hatayı dışlamaya yetmeyebilir).
5. **Çapraz doğrulama şartı:** sonuç ancak şu ikisinden biri sağlanırsa "doğrulanmış" sayılır: (a) ≥2 kırpım birebir aynı okumada anlaştı, (b) tek okuma ama güveni ≥ %80. İkisi de yoksa sonuç üretilir ama CSV'ye yazılmaz.

### Adım 7 — CSV'ye yazma (`log_result`)
```python
if not (result['valid'] and result['cross_validated'] and final_conf >= 0.35):
    return None   # kalite kapısından geçemedi, yazılmaz
```
Ayrıca **yakın-eşleşme tekrar önleme**: yeni okunan plaka, son `DEDUP_WINDOW=120` saniye içinde zaten loglanmış bir plakayla ya birebir aynıysa ya da tek karakter farkla aynıysa (hayalet karakter ekleme/düşme durumu), tekrar yazılmaz — sadece o kaydın "son görülme" zamanı güncellenir.

Yazma anında iz `resolved=True` işaretlenir — bundan sonra o iz için bir daha **hiç** OCR çalışmaz (yazılsın ya da dedup'a düşsün fark etmez, cevap zaten bilinir).

### Adım 8 — İz kapanması
```python
if sample_idx - tr.last_seen > MISS_LIMIT:   # 15 örnek boyunca görülmedi
    if not tr.resolved:
        result = attempt_read(tr, ..., "İz kapandı")   # son bir şans
        log_result(...)
```
Bir iz 15 örnek boyunca hiçbir kutuyla eşleşmezse (araç kadrajdan çıktı), kapanır. Henüz çözülmemişse son bir OCR denemesi yapılır — bu, "artık kaybolacak, elimde ne varsa değerlendireyim" mantığı.

**Video/yayın bitişi de aynı muameleyi görür:** döngü sona erdiğinde hâlâ açık kalan (ve çözülmemiş) tüm izler için aynı son deneme yapılır — yoksa video kısa kesildiğinde tam önünüzdeki bir plaka hiç işlenmeden kaybolurdu.

---

## 5. Zaman mı, sayı mı? — tam liste

| Mekanizma | Birim | Neden bu birim |
|---|---|---|
| Dosya modu örnekleme | kare indeksi | tekrarlanabilirlik — PC hızından bağımsız hep aynı kareler seçilsin |
| Canlı mod örnekleme | saniye (duvar saati) | güncellik — kaynağı kontrol edemiyoruz, geriye düşmemek için zamana bakmalıyız |
| Kırpım saklama aralığı (`CROP_GAP`) | saniye | "bağımsız kanıt" gerçek dünya zamanına göre tanımlı, kaynağın fps'inden bağımsız olmalı |
| Kırpım hafızası (`KEEP_CROPS`) | sayı | "en iyi kaç tane" sorusu, zamanla ilgisi yok |
| OCR tetikleme eşiği | sayı (kırpım) | "yeterli kanıt birikti mi" sorusu, zaman değil |
| İz kapanması (`MISS_LIMIT`) | **örnek sayısı** | ama gerçek saniye karşılığı moda göre değişir (bkz. aşağıda) |
| Toplam işleme süresi (`max_seconds`) | saniye | kullanıcının "ne kadar süre izlensin" isteği doğrudan zamanla ifade edilir |
| Tekrar loglama penceresi (`DEDUP_WINDOW`) | saniye | "kaç dakika içinde" sorusu gerçek zamana ait |

**`fps` değişkeni kendi başına bir karar mekanizması değil** — sadece dosya modunda kare-indeksini saniyeye çevirmek için kullanılan bir birim dönüştürücü (`video_sec = frame_no / fps`). Canlı modda hiç kullanılmıyor çünkü zaten `time.time()` ile doğrudan saniye ölçülüyor.

**`MISS_LIMIT`'in gizli yan etkisi:** 15 bir **sayı**, ama hissettiğin şey saniye. Dosya modunda 60fps kaynak saniyede 10 örnek verir (60÷6) → 15 örnek = 1.5 sn. Canlı modda sabit 4 Hz → 15 örnek = 3.75 sn. Aynı "15" sayısı, canlı modda izi daha sabırlı (daha uzun) açık tutuyor — bu bilinçli bir tasarım değil, örnekleme hızı farkının doğal sonucu.

---

## 6. Dosya modu vs canlı mod — neden ayrıldılar

| | Dosya modu | Canlı mod |
|---|---|---|
| Öncelik | **Doğruluk / tekrarlanabilirlik** | **Güncellik / gecikmesizlik** |
| Örnekleme | kare indeksi (`% FRAME_SKIP`) | zaman (`SAMPLE_PERIOD`) |
| Tampon | önemsiz (dosya sabit duruyor) | `BUFFERSIZE=1` — eski kare birikmesin |
| Riski | yok — dosya bizi beklemez | işleme yavaş kalırsa görüntü geriye düşer |

Tek cümlede: dosyada zaman bizim elimizde (istediğimiz hızda okuruz), canlıda zaman kaynağın elinde (o gönderir, biz yetişmeye çalışırız). Bu yüzden dosya modu "hep aynı sonucu ver" için kare sayar, canlı mod "hep güncel kal" için saati kontrol eder.

**Geri kalan her şey (Adım 2-8) iki modda da birebir aynı** — YOLO, iz eşleştirme, kırpım biriktirme, OCR tetikleme, çapraz doğrulama, CSV yazma, iz kapanması. Sadece "hangi ham kare pipeline'a girsin" kararı farklı; pipeline'ın kendisi tek.

---

## 7. Uçtan uca somut örnek (canlı mod, tek plaka)

Bir araba kadraja girdi, 5 saniye görünüp çıktı:

| Video saniyesi | Olay |
|---|---|
| 0.00 | Örnek alındı, YOLO kutu buldu, hiçbir izle örtüşmüyor → **iz#1 açıldı**, ilk kırpım kesildi (kırpım: 1) |
| 0.25 | Yeni örnek, kutu iz#1 ile örtüşüyor → iz güncellendi. Son kırpımdan 0.25 sn geçti (`< CROP_GAP=0.4`) → ayrı kırpım açılmadı, kalitelisi tutuldu (kırpım: 1) |
| 0.50 | Aradan 0.5 sn geçti (`> 0.4`) → yeni kırpım slotu açıldı (kırpım: 2) |
| 1.00 | Kırpım sayısı 3'e ulaştı → **ilk OCR denemesi**. 3 kırpım da okundu, ikisi `34 ABC 123` dedi → **çapraz doğrulama sağlandı** → CSV'ye yazıldı, iz#1 `resolved=True` |
| 1.00–5.00 | İz zaten çözüldü, artık hiç OCR çalışmıyor — sadece ucuz YOLO+iz takibi devam ediyor, ekran akıcı |
| 5.00 | Araba çıktı, kutu bulunamıyor. 15 örnek (≈3.75 sn) boyunca eşleşmezse iz kapanır — ama zaten çözülmüş olduğu için kapanışta ek bir şey olmaz |

**Kötü senaryo farkı:** 1.00 sn'de kırpımlar anlaşmasaydı (üçü de farklı okusa, hiçbiri %80'i geçmese), sonuç yazılmazdı; iz kırpım biriktirmeye devam ederdi, 2.0 sn'de (5 kırpımla) ve 3.0 sn'de (7 kırpımla) tekrar denenirdi. Araba çözülmeden çıkıp gitseydi, iz kapanırken son bir "son şans" denemesi yapılırdı.

---

## 8. Sık karışan noktalar — doğrudan cevap

**"3 saniyede kaç kez YOLO çalışır?"**
`3 sn × örnekleme hızı`. Canlı modda örnekleme hızı = `min(kaynak fps, 4 Hz)` — yani kaynak 12fps de olsa 60fps de olsa, 4Hz'i geçemez, cevap hep ~12. Dosya modunda `(3 × dosya fps) / 6` — 24fps dosyada 12, 60fps dosyada 30.

**"Kamera 2 fps'lik yavaş bir şeyse ne olur?"**
`SAMPLE_PERIOD=0.25` (4 Hz) sadece **hızlı kaynağı yavaşlatır**, yavaş kaynağı hızlandıramaz. 2 fps'te gelen her kare zaten 4 Hz'den yavaş geldiği için hiçbiri atlanmaz, hepsi işlenir — efektif hız kaynağın kendi hızına (2 Hz) eşitlenir.

**"Tampon 1 kare ne demek?"**
OpenCV canlı akışta gelen kareleri bir kuyrukta biriktirir. İşleme kaynaktan yavaş kalırsa kuyrukta eski kareler birikir, ekranda gördüğün gerçek zamandan geriye düşer. `BUFFERSIZE=1` bu kuyruğu 1 kareye sabitler — yeni kare gelince eskisi atılır, hep güncele yakın kalınır (ama aradaki kareler feda edilir, hiç işlenmez).

**"Neden bazen ekran tamamen duruyor?"**
Kodda bilinçli bir `pause` yok. Bir iz yeterli kırpım biriktirdiğinde `attempt_read` tetiklenir ve bu, aynı döngü içinde **senkron/bloklayan** şekilde çalışır (5 kırpım × ~10 OCR çağrısı = ~50 işlem). O sürede ekrana yeni kare basılmaz. OCR bitince akış devam eder. Bu, PC gücü yetersizliği değil, OCR'ın işleme döngüsüyle aynı thread'de çalışmasının doğal sonucu.

**"Baret/yelek tespitinde bu iz/kırpım/OCR mantığı var mı?"**
Hayır. Baret/yelek modeli her karede doğrudan sınıflandırma yapıyor (hardhat / no-hardhat / safety-vest / no-safety-vest / person) — okunacak bir yazı yok, tek karenin cevabı zaten nihai cevap. Bu yüzden o tarafta iz takibi, kırpım biriktirme, oylama gibi bir katman yok; sadece YOLO çalışır ve sonuç doğrudan CSV'ye yazılır.
