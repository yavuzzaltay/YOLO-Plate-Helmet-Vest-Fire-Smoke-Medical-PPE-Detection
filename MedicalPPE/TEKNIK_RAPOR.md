# MedicalPPE — Teknik Rapor

Tıbbi ortamda (ameliyathane, klinik) çalışanların KKD (kişisel koruyucu
ekipman: eldiven, bone, önlük, maske, gözlük, yüz kalkanı, galoş, forma)
kullanımını görüntü/videodan denetleyen modül. Bu belge modülün
**mevcut** teknik yapısını anlatır; geçmiş kararların gerekçeleri ve
ölçüm tarihçesi için `README.md`, tüm oturumun anlatısı için kök
dizindeki `PROJE_RAPORU.txt` içindeki 5. bölüme bakılabilir.

## 1. Model

- Mimari: **YOLO11s** (Ultralytics), 14 sınıf, tek aşamalı nesne tespiti.
- Ağırlık dosyası: `MedicalPPE.pt` (640×640 girişle eğitildi; "best.pt"
  gibi genel-geçer bir ad yerine modül adını taşıyor). Yedek kopya
  `best_v1_640px_2026-07-16.pt`.
- `find_weights()` önce kök dizindeki `MedicalPPE.pt`'yi arar, yoksa
  `runs/detect/*/weights/best.pt` altındaki (henüz adlandırılmamış, ham
  Ultralytics çıktısı) en yeni eğitime düşer.
- 14 sınıf, **pozitif+negatif çift şema**: her ekipman için hem "takılı"
  hem "takılı değil/yanlış takılı" sınıfı var (`surgical-gloves` /
  `no-surgical-gloves` gibi). Bu şema, modelin "eksiklik"i doğrudan
  tahmin etmesini sağlar; kod tarafında kişi-ekipman eşleştirmesi
  gerektirmez (bkz. §4 "direct" mod).

## 2. Veri hattı (`prepare_dataset.py`)

3 çekirdek Roboflow projesi (`ds1`-`ds3`, best.pt'yi eğiten Kaggle
akışıyla birebir aynı kaynaklar) + 2 takviye proje (`ds4`-`ds5`,
zayıf negatif sınıflara gerçek-ortam örneği eklemek için) tek bir
`dataset/` altında birleştirilir:

- Her projenin kendi `data.yaml`'ındaki sınıf adları okunur, isimce
  (büyük/küçük harf farkı dahil) sabit **14 elemanlı `FINAL_NAMES`**
  sırasına eşlenir; etiket dosyalarındaki class id'ler bu ortak sıraya
  göre yeniden yazılır (id'lere körlemesine güvenilmez).
- `ds4`/`ds5` için `only_final_ids={2,4,10}`: yalnızca zayıf sınıfları
  içeren görüntüler alınır (binlerce alakasız görüntüyle şişirmemek
  için).
- `ds4`/`ds5` için `test_to_train=True`: bu kaynakların hiçbir görüntüsü
  test split'ine girmez — kendi test split'leri de train'e akar. Böylece
  `ds1`-`ds3`'ün orijinal test görüntüleri (535 adet) **sabit bir
  benchmark** olarak kalır; yeniden eğitim sonrası karşılaştırmalar
  elmayla elma olur.
- Windows'un 260 karakter `MAX_PATH` sınırı yüzünden ham indirme kısa
  bir yola (`C:\ppe_raw`) yapılır, yalnızca birleştirme sonrası kısa/
  sıralı adlarla proje klasörüne kopyalanır.
- API anahtarı ortam değişkeni veya `.roboflow_key` dosyasından okunur
  (repo public olduğu için asla koda gömülmez, `.gitignore`'da).

## 3. Sınıf adı çözümleme (`ITEM_ALIASES`, `resolve_model_classes`)

Model hangi veri setiyle eğitilirse eğitilsin sınıf adları değişebilir
("gloves", "Glove", "no_glove", "MaskOff"...). Kod, `model.names`'i
okuyup her sınıfı çalışma zamanında çözer:

1. Ad normalize edilir (küçük harfe çevrilir, Türkçe karakterler
   ASCII'ye indirgenir, alfasayısal olmayanlar atılır).
2. `PERSON_ALIASES` içinde mi diye bakılır ("person", "worker"...).
3. `ITEM_ALIASES` tablosundaki 8 kanonik ekipmanla (eldiven, bone,
   önlük, forma, galoş, gözlük, maske, siperlik, yüz-koruması,
   medikal-kıyafet) doğrudan eşleşme aranır → **pozitif** sınıf.
4. Eşleşmezse `NEGATION_PREFIXES` ("no", "non", "without", "incorrect",
   "wrong", "improper", "missing") veya "off" soneki soyulup kalan kısım
   tekrar denenir → **negatif** sınıf.
5. Hiçbiri tutmazsa sınıf `"unknown"` kalır: ekranda çizilir/sayılır ama
   uyum mantığına (ihlal kararına) girmez — yanlış varsayımdansa dışarıda
   bırakmak tercih edildi.

Bu çözümleme sayesinde kodun geri kalanı sınıf adlarına değil, kanonik
`ClassInfo(role, item, positive)` üçlüsüne göre çalışır; yeniden eğitim
sınıf adlarını değiştirse bile kod değişmeden çalışır.

## 4. Video/canlı izleme hattı (`MedicalPPEMonitor`)

Kare akışı 4 katmandan geçer (mimari `FireAndSmokeVideo.FireSmokeMonitor`
ile paralel):

```
kare ──▶ [1] YOLO track (ByteTrack, persist=True)
     ──▶ [2] sınıf çözümleme + sınıf-bazlı eşik (DEFAULT_THRESHOLDS)
     ──▶ [3] mod seçimi + gözlem çıkarımı
     ──▶ [4] N-of-M zamansal onay (kayan pencere)
     ──▶ [5] onay anında kanıt kaydı (fotoğraf + CSV, tek seferlik)
```

**Mod seçimi** modelin ürettiği sınıflara göre otomatik:
- **`direct`** — model negatif sınıflar (`no_*`) üretiyorsa (bu modelin
  durumu budur): her negatif tespit doğrudan ihlal adayıdır, kişi-ekipman
  eşleştirmesi gerekmez.
- **`person`** — yalnızca kişi + pozitif ekipman varsa: ekipman kutuları
  `_find_owner()` ile kişilere atanır (kutu merkezi, kişi kutusunun
  %15 genişletilmiş sınırları içindeyse eşleşir; birden fazla aday varsa
  merkeze en yakın kişi kazanır); kişide eksik kalan zorunlu ekipman
  ihlal adayı olur.
- **`presence`** — ne kişi ne negatif sınıf varsa: yalnızca tespit/sayım,
  ihlal çıkarımı yapılmaz.

**N-of-M zamansal onay**: tek karelik "eldiven yok" güvenilmez (el
cebe girer, gölgede kalır, ten rengine karışır). Her iz için
`deque(maxlen=M)` bir kayan pencere tutar; pencerede en az `N` isabet
varsa ihlal **onaylanır**. Varsayılan `(6, 10)` — yangın modülündeki
`(4,8)/(5,10)`'dan daha temkinli, çünkü oklüzyon riski daha yüksek.
Kişi bir karede hiç görünmediyse pencereye 0/1 **yazılmaz** (yangının
"ıska=0" mantığından farklı — burada kişiyi haksız yere aklamamak için).
İz `TRACK_FORGET_LIMIT=12` kareden fazla kaybolursa silinir.

**Çakışma bastırma** (`_suppress_conflicts`): aynı ekipmanın pozitif ve
negatif tespiti aynı bölgeye düşerse (IoU>0.45), NMS bunu çözemez
(farklı sınıflar) — kod düşük güvenli olanı eler.

**Kanıt kaydı**: onaylanan her (iz, eksik ekipman) çifti için bir kez
anotasyonlu JPEG (`IhlalKayitlari/`) ve bir CSV satırı
(`medikal_ihlal_log.csv`: tarih, video sn, iz id, eksik ekipman, güven,
dosya adı) yazılır. Alarm/bildirim yok — sistemin görevi kanıt biriktirmek.

## 5. Fotoğraf modu (`process_photo`) — TTA + dilimli tespit

Tek karede zamansal onay uygulanamaz; bunun yerine isabeti artırmak için
iki teknik eklendi:

**a) TTA (Test-Zamanı Veri Artırma)** — `model.predict(..., augment=True)`
her zaman açık. Video/canlı akışta hız kritik olduğu için kapalı.

**b) Dilimli tespit (SAHI yaklaşımı)** — küçük ihlal bölgelerini
(çıplak el, korumasız yüz) yakalamak için:

```
              ┌─────────────────────────────┐
tam kare 640px│  büyük nesneler (kişi, önlük)│──┐
              └─────────────────────────────┘  │
                                                ├──▶ tekilleştirme ──▶ eşik + boyut
  2×2 dilim,   ┌──────┬──────┐                 │       filtresi ──▶ çakışma bastırma
  %20 örtüşme, │  A   │  B   │  768px'te ───────┘
  her biri     ├──────┼──────┤  modele (küçük/
  768px'te     │  C   │  D   │  kesik/pozitif
  modele       └──────┴──────┘  filtrelenir)
```

Dilim geçişinden yalnızca şu üç ölçülmüş filtreden geçen kutular kabul
edilir:
1. **Boyut filtresi** — kutu alanı tam kare alanının %5'inden büyükse
   atılır (büyük nesne zaten tam karede yakalanmıştır; dilimde büyük
   çıkan kutu genelde yarım-gövde yanlış alarmıdır).
2. **Kenar filtresi** — kutu, dilimin *iç* kenarına değiyorsa atılır
   (nesne kesilmiş demektir; kare sınırındaki kenarlar hariç).
3. **Yalnız-negatif filtresi** — yalnızca `no_*` (ihlal) sınıfları kabul
   edilir; pozitif ekipman sınıfları tam kareden zaten güçlü geliyor.

Tekilleştirme (`_merge_duplicates`) **tek yönlüdür**: tam kare kutularına
asla dokunulmaz, dilimden gelen kutu yalnızca eklenir ya da (aynı yerde
zaten bir kutu varsa) güveni daha yüksekse onun yerine geçer. Aynı
kutu sayılma kuralı: IoU>0.55 **veya** küçük kutunun %80'i büyüğün
içinde (kesik-yarım kutu, tam kutuyla düşük IoU verir ama içinde kalır).

Dilim geçişinin model giriş boyutu (`TILE_IMGSZ=768`), tam kareninkinden
(640) **kasıtlı olarak farklı** — bu ayrım ölçülerek seçildi (bkz. §7).

## 6. Eşik kalibrasyonu — iki ayrı set

Video ve fotoğraf hatları farklı güven dağılımı ürettiği için (TTA
skorları sistematik yükseltiyor) **iki ayrı eşik seti** var, ikisi de
`ap_per_class`'ın F1-Confidence eğrisinden (her sınıf için F1'in zirve
yaptığı güven noktası) kalibre edildi:

| Set | Kullanan hat | Kalibrasyon kaynağı |
|---|---|---|
| `DEFAULT_THRESHOLDS` | Video/canlı (`MedicalPPEMonitor`) | düz `model.val()` |
| `PHOTO_THRESHOLDS` | Fotoğraf (`process_photo`) | gerçek hat (TTA+dilim) F1 eğrisi |

`PHOTO_THRESHOLDS` ortalama 0.15 daha yüksek — TTA'nın güveni
sistematik yükseltmesinin doğrudan sonucu. Yeniden kalibrasyon:
`evaluate.py` (video seti) / `MedicalPPEVideo.evaluate_pipeline()`
(fotoğraf seti, F1 tepe noktalarını basar).

## 7. Ölçülen çıkarım-zamanı deneyler

Aynı sabit 535 görüntülük test setinde, **yeniden eğitim olmadan**,
salt çıkarım-zamanı parametrelerle 3 deney (`ap_per_class` + `box_iou`
ile Ultralytics'in kendi AP matematiğiyle ölçüldü):

| Deney | Sonuç | Karar |
|---|---|---|
| Tam kareyi 768/960'a çıkarmak | küçük sınıflar +3..+7, büyük sınıflar -3..-7, genel mAP50-95 düşüyor (ölçek uyuşmazlığı, eğitim 640'taydı) | ❌ Red |
| Dilim ızgarasını 3×3 yapmak | 640px kaynakta 246px dilim oluyor, 2.6× büyütme bulanıklığa dönüşüyor, genel mAP50 düşüyor, süre 2 kat | ❌ Red |
| Dilimleri 768'de koşmak (tam kare 640 kalırken) | genel mAP50 +0.2 puan, en zayıf sınıf (no-surgical-gloves) +2.0 puan, pozitifler birebir korundu | ✅ Kabul (`TILE_IMGSZ=768`) |

Ayrıca önceki turda (2026-07-17) filtresiz dilimleme denenip
reddedilmişti (person mAP50 0.88→0.25'e çökmüştü); üç filtre
(boyut/kenar/yalnız-negatif) o denemeden çıktı.

## 8. Model kıyaslama disiplini

Bir alternatif model (`best_model_v2`, 2026-07-22) denenip **reddedildi**:
mAP50-95'te 14 sınıfın 13'ünde kötüydü, recall 10 sınıfta düşmüştü,
üstelik alakasız bir sahnede (yangın fotoğrafı) sahte "eldiven yok"
alarmı üretiyordu. Bu olay standart bir prosedür doğurdu: **model
kıyaslamasında tek metriğe (özellikle sadece mAP50-95'e) bakılmaz** —
4 metrik (Precision/Recall/mAP50/mAP50-95) × sınıf tablosu + alakasız
sahne sağlamlık testi zorunlu adımlar oldu. Tam kıyas tablosu
`README.md`'de.

## 9. Arayüz entegrasyonu (`app.py`)

- `🧑‍⚕️ Tıbbi PPE` sekmesi, `MedicalPPEVideo.process_photo` (Görsel) ve
  `MedicalPPEMonitor` (Video, dosya/IP kamera) üzerinden çalışır.
- "📊 Metrikleri Hesapla" butonu `MedicalPPEVideo.evaluate_pipeline()`'ı
  çağırır — düz `model.val()` **değil**, kullanıcının gerçekten
  çalıştırdığı hattı (TTA+dilim) test setinde ölçer, ilerleme çubuğuyla.
  Sonuç Precision/Recall/F1/mAP50/mAP50-95 + sınıf bazlı açılır tablo
  olarak gösterilir.

## 10. Bilinen sınırlamalar / bekleyen iş

- Model hâlâ 640px'te eğitildi; zayıf negatif sınıfların çekirdek
  verisi çoğunlukla tek stüdyo kaynağından geliyor.
- Hazır ama **koşulmamış** bir yeniden eğitim var: `train.py`
  (imgsz 768, batch 12 — VRAM ölçümlü, 14.375 görüntü, zayıf sınıflara
  10-15× veri, sabit test benchmark'ı korunarak). Bu, çıkarım-zamanı
  ayarların ulaşabildiği tavanın ötesinde bir kazanç potansiyeli taşıyor
  ama çok saatlik GPU işi olduğu için kullanıcı onayı gerektiriyor.
- `C:\ppe_raw` altındaki ham (birleştirilmemiş) veri hâlâ diskte.

## Dosya haritası

| Dosya | Görev |
|---|---|
| `MedicalPPEVideo.py` | İzleme çekirdeği: `MedicalPPEMonitor` (video), `process_photo` (fotoğraf), `evaluate_pipeline` (gerçek hat ölçümü). Arayüz içermez. |
| `prepare_dataset.py` | 5 Roboflow projesini indirip `dataset/` altında birleştirir. |
| `train.py` | Sonraki eğitim şablonu (yolo11s, AdamW, 768px, batch 12). |
| `evaluate.py` | Video eşikleri için F1-zirve önerisi + test seti metrikleri. |
| `evaluate_kaggle.ipynb` | `evaluate.py`'nin Kaggle'da tek başına çalışan hali. |
| `MedicalPPE.pt` / `best_v1_640px_2026-07-16.pt` | Aktif model + yedek. |
| `IhlalKayitlari/`, `medikal_ihlal_log.csv` | Video hattının kanıt fotoğrafları ve olay kaydı. |
