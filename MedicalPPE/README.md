# MedicalPPE — Tıbbi PPE (KKD) Uyum İzleme

Sağlık ortamında çalışanların tıbbi koruyucu ekipmanları — **eldiven, bone,
önlük, gözlük** (model destekliyorsa maske/siperlik de) — takıp takmadığını
görüntü ve videodan denetler. Streamlit arayüzündeki (app.py) "Tıbbi PPE"
sekmesi bu modülü kullanır.

> **Durum:** Aktif model **v1** (`MedicalPPE/best.pt`, 2026-07-16), sekme
> aktif, eşik kalibrasyonu güncel. 2026-07-22'de denenen alternatif model
> ("v2") aynı sabit test setinde 14 sınıfın 13'ünde daha kötü çıktığı için
> **reddedildi ve devreye ALINMADI** — ayrıntılı kanıt tablosu aşağıda
> "Denenen ve reddedilen: best_model_v2" bölümünde.

## Model kimliği (best.pt) — v1, 2026-07-16 (AKTİF)

- **yolo11s**, 100 epoch, imgsz 640, batch 16 (Kaggle, 2026-07-16)
- Eğitim sonu (Kaggle val) doğrulama metrikleri: mAP50 0.771 · mAP50-95 0.444 · P 0.751 · R 0.750
- **Bağımsız yerel test setinde** (535 görüntü, `prepare_dataset.py` ile aynı 3
  Roboflow projesinden indirilip birleştirildi — modelin hiç görmediği görüntüler):
  **mAP50 0.800 · mAP50-95 0.479 · P 0.763 · R 0.803**
- 14 sınıf — pozitif + negatif şema sayesinde izleme **direct** modda çalışır:

| Ham sınıf | Kanonik | Kutup |
|---|---|---|
| person | kişi | — |
| surgical-cap / no-surgical-cap | bone | + / − |
| surgical-gloves / no-surgical-gloves | eldiven | + / − |
| surgical-mask | maske | + |
| surgical-gown, coverall | onluk | + |
| surgical-scrubs | forma | + |
| shoe-covers | galos | + |
| face-shield | siperlik | + |
| goggles | gozluk | + |
| no-facial-gear | yuz-korumasi | − |
| no-medical-attire | medikal-kiyafet | − |

## Denenen ve REDDEDİLEN: best_model_v2 (2026-07-22)

Kullanıcı tarafından "best_model_v2.zip" olarak sağlanan alternatif bir
model denendi (Kaggle, aynı ayarlar: yolo11s, 100 epoch, imgsz 640,
batch 16 — hangi veri/seed farkının sonucu değiştirdiği bilinmiyor;
bizim planladığımız 768px+genişletilmiş-veri koşusu DEĞİL, train.py
hâlâ koşulmadı). AYNI sabit 535 görüntülük test setinde ölçüldü ve
**14 sınıfın 13'ünde mAP50-95 gerilediği için reddedildi.**

Genel özet (v1 → v2):

| Metrik | v1 | v2 | Fark |
|---|---|---|---|
| Precision | 0.763 | 0.767 | +0.004 |
| Recall | 0.803 | 0.763 | **−0.040** |
| mAP50 | 0.800 | 0.786 | **−0.014** |
| mAP50-95 | 0.479 | 0.464 | **−0.015** |

Sınıf bazlı tam kıyaslama (v1 → v2, tüm 14 sınıf × 4 metrik):

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| person | 0.868→0.859 ↓ | 0.871→0.856 ↓ | 0.917→0.897 ↓ | 0.607→0.592 ↓ |
| surgical-cap | 0.880→0.881 ≈ | 0.908→0.897 ↓ | 0.916→0.913 ↓ | 0.545→0.536 ↓ |
| no-surgical-cap | 0.585→0.615 ↑ | 0.697→0.544 ↓↓ | 0.598→0.573 ↓ | 0.253→0.235 ↓ |
| surgical-gloves | 0.738→0.738 ≈ | 0.788→0.768 ↓ | 0.806→0.779 ↓ | 0.479→0.453 ↓ |
| no-surgical-gloves | 0.502→0.485 ↓ | 0.532→0.444 ↓↓ | 0.441→0.418 ↓ | 0.195→0.174 ↓ |
| surgical-mask | 0.867→0.869 ≈ | 0.907→0.909 ≈ | 0.907→0.914 ↑ | 0.539→0.531 ↓ |
| surgical-gown | 0.798→0.821 ↑ | 0.869→0.875 ≈ | 0.899→0.905 ↑ | 0.610→0.594 ↓ |
| shoe-covers | 0.717→0.738 ↑ | 0.705→0.716 ↑ | 0.730→0.749 ↑ | 0.294→0.300 ≈ |
| surgical-scrubs | 0.741→0.685 ↓ | 0.819→0.831 ↑ | 0.806→0.775 ↓ | 0.509→0.480 ↓ |
| face-shield | 0.862→0.854 ↓ | 0.923→0.923 ≈ | 0.943→0.937 ↓ | 0.604→0.592 ↓ |
| no-facial-gear | 0.621→0.609 ↓ | 0.710→0.653 ↓ | 0.617→0.595 ↓ | 0.304→0.287 ↓ |
| no-medical-attire | 0.743→0.754 ↑ | 0.740→0.652 ↓↓ | 0.776→0.748 ↓ | 0.455→0.444 ↓ |
| goggles | 0.898→0.928 ↑ | 0.872→0.829 ↓ | 0.905→0.905 ≈ | 0.602→0.582 ↓ |
| coverall | 0.859→0.900 ↑ | 0.904→0.785 ↓↓ | 0.935→0.903 ↓ | 0.715→0.695 ↓ |

Metrik başına kaç sınıfta iyileşti/kötüleşti (14 sınıf üzerinden):
Precision 8 iyi / 5 kötü / 1 eşit — Recall 3 iyi / 10 kötü / 1 eşit —
mAP50 3 iyi / 10 kötü / 1 eşit — **mAP50-95: 1 iyi / 13 kötü.**
Yani precision'daki hafif kazanç dışında v2, hemen her sınıfta ve
hemen her metrikte v1'den geride — özellikle recall'da bazı sınıflarda
(no-surgical-cap, no-medical-attire, coverall) 9-15 puanlık büyük düşüşler var.

Ayrıca gerçek görüntüyle doğrulamada bir sağlamlık sorunu da bulundu: v1'in
"alakasız sahnede (yangın karesi, insan/tıbbi ortam YOK) sıfır yanlış
alarm" davranışı v2'de bozuluyordu — `no-surgical-gloves` sınıfının
F1-zirvesi eşiği v2'de çok düşük (0.10) çıktığı için, o sınıf tamamen
alakasız bir yangın fotoğrafında bile düşük güvenle (~0.13-0.16)
"eldiven yok" diye tetikleniyordu.

**Karar: v2 reddedildi, best.pt v1 olarak kaldı.** `best_v1_640px_2026-07-16.pt`
dosyası (v1 ile birebir aynı içerik) yedek olarak duruyor; `MedicalPPEVideo.py`
içindeki `DEFAULT_THRESHOLDS` de v1 değerlerinde. v2'nin ham checkpoint'i
saklanmadı (repo'da yer kaplamasın diye silindi) — gerekirse kullanıcıdan
tekrar istenebilir.

## Eşik kalibrasyonu — TAMAMLANDI (v1)

`DEFAULT_THRESHOLDS` (`MedicalPPEVideo.py`) 2026-07-17'de `evaluate.py`'nin
yerel test setinde (bkz. yukarı) ürettiği F1-Confidence zirvesiyle dolduruldu:

```
person 0.36 · surgical-cap 0.43 · no-surgical-cap 0.26 · surgical-gloves 0.43
no-surgical-gloves 0.37 · surgical-mask 0.34 · surgical-gown 0.49 · shoe-covers 0.21
surgical-scrubs 0.36 · face-shield 0.46 · no-facial-gear 0.25 · no-medical-attire 0.21
goggles 0.36 · coverall 0.18
```

En zayıf sınıflar (F1'e göre): `no-surgical-gloves` (0.53), `no-surgical-cap`
(0.66), `no-facial-gear` (0.69) — hepsi **negatif** sınıflar, tıpkı Sensors
2025 çalışmasının işaret ettiği gibi "yokluk" tespiti "varlık" tespitinden
daha zor. Veri büyütmede önceliklendirilecek sınıflar bunlar (bkz. Yol haritası).

Yeniden kalibrasyon gerekirse (yeni eğitim, yeni veri):
1. `python prepare_dataset.py` — best.pt'yi eğiten Kaggle akışıyla AYNI 3
   Roboflow projesini indirip `MedicalPPE/dataset/` altında birleştirir
   (`RAW_DIR` olarak `C:\ppe_raw` kullanır — Windows'un 260 karakter yol
   sınırını aşmamak için proje klasörünün dışında, kısa bir yol). API
   anahtarı gerekir: `ROBOFLOW_API_KEY` ortam değişkeni ya da
   `.roboflow_key` dosyası (ikisi de `.gitignore`'da, commit'e girmez).
2. `python evaluate.py` → yapıştırmaya hazır `DEFAULT_THRESHOLDS` basar.

Kaggle'da çalıştırmak istersen `evaluate_kaggle.ipynb` da hâlâ mevcut
(best.pt ile data.yaml'ı kendisi bulur, test split yoksa val'e düşer).

## Dosyalar

| Dosya | Görev |
|---|---|
| `MedicalPPEVideo.py` | İzleme çekirdeği: `MedicalPPEMonitor` (video) + `process_photo` (tek kare). Arayüz içermez. |
| `train.py` | Sonraki eğitimler için şablon (yolo11s, AdamW, erken durdurma). |
| `evaluate.py` | Test seti metrikleri **+ sınıf bazlı F1-zirvesi eşik önerisi** (aşağıya bak). |
| `evaluate_kaggle.ipynb` | evaluate.py'nin Kaggle'da tek başına çalışan hali (veri seti Kaggle'daysa bunu kullan). |
| `prepare_dataset.py` | 3 Roboflow projesini indirip `dataset/` altında birleştirir (best.pt'yi eğiten Kaggle akışının yerel karşılığı). |
| `IhlalKayitlari/` | Onaylanan ihlallerin anotasyonlu kanıt fotoğrafları (otomatik oluşur). |
| `medikal_ihlal_log.csv` | Olay kaydı: tarih, video sn, iz id, eksik ekipman, güven, kanıt dosyası. |

## Mimari (FireAndSmoke ile aynı katman düzeni)

1. **YOLO track** — ByteTrack kişilere/tespitlere kalıcı kimlik verir.
2. **Sınıf çözümleme** — model eğitimde olduğu için sınıf adlarını şimdiden
   bilmiyoruz; `resolve_model_classes()` model.names'i takma ad tablosuyla
   (`ITEM_ALIASES`) kanonik ekipmanlara eşler (`gloves`→eldiven,
   `no_gown`→önlük ihlali, `person`→kişi...). Eğitim hangi adlarla biterse
   bitsin kod değişmez; eşleşmeyen ad çıkarsa tabloya bir satır eklemek yeter.
3. **Uyum mantığı** — modelin sınıflarına göre otomatik mod seçimi:
   - `direct`: modelde `no_glove` gibi **negatif sınıflar** varsa her negatif
     tespit doğrudan ihlal adayı (literatürün önerdiği, en sağlıklı düzen).
   - `person`: model yalnızca kişi + pozitif ekipman veriyorsa ekipman
     kutuları kişilere atanır (merkez, %15 genişletilmiş kişi kutusunda mı?);
     kişide eksik kalan zorunlu ekipman ihlal adayı olur.
   - `presence`: kişi de negatif sınıf da yoksa yalnızca tespit/sayım.
4. **N-of-M zamansal filtre** — varsayılan **6/10**: son 10 örnekleme
   karesinin ≥6'sında ihlal görülürse onay. Yangının 4/8'inden temkinli,
   çünkü el/kol oklüzyonu tek karelik sahte "eldiven yok" üretebilir.
   Kişi bir karede hiç görünmediyse pencereye 0 da 1 de yazılmaz (bilgi yok).
5. **Kanıt kaydı** — onay anında anotasyonlu kare + CSV satırı; aynı iz +
   aynı eksik ekipman için yalnızca bir kez. Alarm/bildirim bilinçli yok.

## Hızlı test

- `python MedicalPPEVideo.py 0` → webcam'de canlı izleme (konsol modu ve
  sınıf eşleşmesini basar); `python MedicalPPEVideo.py video.mp4` de olur.
- Arayüzden: app.py → 🧑‍⚕️ Tıbbi PPE sekmesi (görsel / video / IP kamera).
- Doğrulanmış davranış (2026-07-17): 14 sınıfın tamamı takma-ad tablosuyla
  çözülüyor, mod=direct; alakasız sahnelerde (araç pazarı, yangın karesi)
  conf 0.10'da bile sıfır yanlış alarm; track hattı GPU'da ~20 FPS.
  Gerçek test görüntülerinde (`dataset/test/images`) pozitif/negatif
  tespitler ve `process_photo`'nun ihlal özeti Streamlit arayüzünde canlı
  doğrulandı (bkz. "Bilinen düzeltmeler").

### Dilimli tespit — fotoğraf modunda küçük-ihlal güçlendirme (2026-07-17)

Yeniden eğitim GEREKTİRMEDEN, salt çıkarım-zamanı kodla ölçülmüş kazanç:
`process_photo` artık tam kare + TTA geçişine ek olarak kareyi %20 örtüşmeli
2×2 dilime bölüp her dilimi ayrı 640px'te modele sokuyor (SAHI yaklaşımı) —
küçük nesne modelin gözünde ~2 kat büyüyor. Üç ölçülmüş filtreyle:

1. dilimlerden yalnızca **küçük** kutular (kare alanının ≤%5'i) alınır,
2. **iç dilim kenarında kesilen** kutular atılır,
3. yalnızca **negatif (ihlal) sınıflar** kabul edilir; çifte düşen kutularda
   güveni yüksek olan kazanır, pozitif sınıflara asla dokunulmaz.

Her filtre, 535 görüntülük sabit test setinde denenerek seçildi (filtresiz
sürüm person mAP50'yi 0.88→0.25'e çökertiyordu!). **Nihai ölçüm:**

| Sınıf | mAP50 düz | mAP50 dilimli | fark |
|---|---|---|---|
| no-facial-gear | 0.585 | **0.684** | **+9.9 puan** |
| no-surgical-cap | 0.553 | 0.591 | +3.8 |
| no-surgical-gloves | 0.419 | 0.455 | +3.6 |
| no-medical-attire | 0.745 | 0.756 | +1.1 |
| tüm pozitif sınıflar | — | — | ±0.000 (birebir korunur) |
| **GENEL** | 0.777 | **0.790** | **+1.3** |

Bedel: fotoğraf başına çıkarım ~0.07s→~0.24s (fotoğraf modunda önemsiz).
Videoda kapalı (zamansal N-of-M zaten kaçan kareyi telafi ediyor, hız
kritik); gerekirse `process_photo(tiled=...)` gibi videoya da opsiyonel
eklenebilir. Not: buradaki "düz" referansı eşik/uyum filtresiz saf tespit
ölçümüdür; app'teki "Metrikleri Hesapla" (model.val) sayılarıyla birebir
karşılaştırma yapılmamalı.

### Bilinen düzeltmeler (2026-07-17)

Gerçek görüntülerle test ederken `process_photo`'da iki hata bulunup
düzeltildi:
1. **Çifte ihlal raporu**: model `direct` modda (no_* sınıfları var) olsa
   bile fonksiyon AYRICA kişi-ekipman kutu eşleştirmesiyle "kadrajda
   görünmeyen her ekipman eksiktir" diye ikinci bir (yanlış) ihlal listesi
   üretiyordu — ör. ayakkabı fotoğrafa hiç girmemiş olsa da "galos eksik"
   diyordu. Artık `process_photo` da `MedicalPPEMonitor` ile AYNI mod
   seçimini yapıyor; `direct` modda yalnızca gerçek negatif tespitler
   ihlal sayılıyor.
2. **Karışan sayaç anahtarı**: `counts` sözlüğünde pozitif ve negatif
   tespit aynı anahtarda toplanıyordu (`eldiven: 2` hem "2 eldiven var"
   hem "1 var+1 yok" anlamına gelebiliyordu). Negatif tespitler artık
   `"eldiven-YOK"` gibi ayrı anahtarla sayılıyor.

## Araştırma notları

### İncelenen çalışma: MEAG-YOLO (Appl. Sci. 2024, 14(11), 4766)

[MEAG-YOLO](https://www.mdpi.com/2076-3417/14/11/4766), **elektrik trafo
merkezlerindeki** PPE'yi (izole eldiven, baret, kolluk, izole çubuk) hedefler;
YOLOv8n'e MSCA (çok ölçekli dikkat) + EC2f (ECA'lı C2f) + ASFF + GhostConv
ekleyerek kesinliği %2.4 artırıp FLOPs'u %7.3 düşürür.

**Karar: doğrudan kullanmıyoruz.** Gerekçeler:
- Veri seti tıbbi değil (trafo merkezi) ve **paylaşılmamış** (2800 görüntü, özel).
- Kazanç marjinal (+2.4 precision); buna karşılık mimari cerrahisi
  Ultralytics'te özel modül tanımı + custom yaml gerektirir, bakım yükü büyük.
  Aynı bütçeyle model boyutunu n→s yapmak veya imgsz'i 640→768 çekmek
  literatürde tutarlı olarak daha fazla kazandırır.

**Yine de taşıdığımız fikirler:** küçük nesneler (eldiven/gözlük) için çok
ölçekli özellik önemli → çözüm olarak yüksek imgsz + fotoğraf modunda TTA;
sınıf başına ayrı güven eşiği; parlaklık/doygunluk/gürültü augmentasyonu
(Ultralytics varsayılanlarında zaten var).

### Asıl faydalandığımız çalışmalar

- **[Sensors 2025 — Hijyen uyumu YOLO kıyaslaması](https://www.mdpi.com/1424-8220/25/19/6140)**
  (31.371 görüntü; glove/no_glove, hairnet/no_hairnet, mask/no_mask/incorrect_mask):
  - **Pozitif + negatif sınıf şeması** en sağlıklı uyum düzeni → modülün
    `direct` modu bu şemaya göre tasarlandı. Kendi veri setini genişletirken
    `no_*` sınıflarını etiketlemeye değer.
  - Sınıf zorluğu sırası: bone/hairnet en kolay (recall %93.8), maske
    ve "yok" sınıfları en zor → eşikleri sınıf bazlı kalibre etmek şart
    (evaluate.py bunu otomatikler).
  - Nano modeller 31k görüntüde ancak %85.7 mAP50'ye ulaşıyor → "en iyi
    modellerden biri" hedefi için **yolo11s + imgsz 768** önerisi (train.py).
  - Veri kaynakları kamuya açık Roboflow projeleri → veri seti büyütme kaynağı.
- **[CPPE-5](https://arxiv.org/abs/2112.09569)** (tıbbi PPE veri seti;
  coverall, face shield, glove, goggles, mask; 1029 görüntü,
  [HuggingFace'te](https://huggingface.co/datasets/yunusskeete/cppe5)):
  önlük (coverall) ve **gözlük** örneği içeren nadir açık set → özellikle
  gözlük recall'u düşük çıkarsa eğitime karıştırılacak ilk kaynak.

### Zayıf sınıf tanısı (2026-07-17) — no-surgical-cap / no-surgical-gloves / no-facial-gear

En zayıf 3 sınıf (mAP50-95: 0.195 / 0.253 / 0.304 — diğerlerinin neredeyse
yarısı) üzerine kök neden analizi yapıldı, **iki somut sorun** bulundu:

1. **Küçük kutu.** Bu 3 sınıfın medyan kutu alanı görüntünün yalnızca
   **%0.9-1.6'sı**. Karşılaştırma: `no-facial-gear` medyan %1.60 iken
   pozitif karşılığı `face-shield` %16.36 — **10 kat daha küçük**. YOLO'nun
   640px'teki özellik haritaları bu boyuttaki nesneleri ayırt etmekte
   zorlanıyor (klasik küçük-nesne sorunu).
2. **Tek kaynak.** `no-surgical-cap`/`no-surgical-gloves`/`no-facial-gear`/
   `no-medical-attire` örneklerinin **tamamı** (1000+ örnek her biri) TEK
   bir Roboflow projesinden geliyor (`ds2`: ecrioobjectdetection1/meppe);
   `ds1`/`ds3` bu sınıflara hiç katkı yapmıyor. Örnekler görsel olarak
   incelendi (8'er kutu/sınıf, çizilerek): ağırlıklı olarak **stüdyo/stok
   fotoğraf** tarzı (tek kişi, düz arka plan, kontrollü ışık) — gerçek
   hastane ortamının (kamera açısı, kalabalık, hareket bulanıklığı) çok az
   temsilcisi var. Test seti de aynı kaynaktan geldiği için mevcut mAP
   sayıları bile gerçek dağılım kaymasını tam yansıtmıyor olabilir.
   **Etiket kalitesi** kontrol edildi — kutular makul/tutarlı, yani sorun
   etiketleme hatası değil.

**Denenip ELENEN yol:** P2 (stride-4) tespit başlığı — küçük nesneler için
klasik bir çözüm ama kurulu Ultralytics sürümü YOLO11 için hazır bir
`yolo11-p2.yaml` sağlamıyor; elle mimari yazmak MEAG-YOLO'da reddettiğimiz
aynı bakım yükünü getirir. Önce ucuz yollar tükenmeden gündeme alınmadı.

### Yol haritası — UYGULANDI (2026-07-17) ve kalan adımlar

**Uygulananlar (yeni eğitim `medical_ppe_yolo11s_768` bu yapılandırmayla
arka planda başlatıldı):**

1. ✅ **imgsz 640→768** — küçük kutu sorununu doğrudan hedefliyor.
2. ✅ **batch 4→12** (DÜŞÜRme değil YÜKSELTme — ölçümle): mini-koşu VRAM
   probları bu GPU'da 768px'te batch=4→2.72GB, 8→3.25GB, 12→4.75GB
   (rahat), 16→6.06GB (sınırda) gösterdi. "6GB'da batch=4 bile riskli"
   eski bilgisi güncel torch/ultralytics ile geçersizmiş. Büyük fiziksel
   batch BatchNorm istatistiklerini de iyileştirir.
3. ✅ **Tek-kaynak bağımlılığı kırıldı**: Sensors 2025'in ham kaynakları
   (GP: `gp-zmz2y/gp-fqlbs`, Final Work: `nku-oyddi/final-work-nyuq2`)
   ds4/ds5 olarak `prepare_dataset.py`'ye eklendi — yalnızca zayıf sınıf
   içeren görüntüler alınır (`only_final_ids`), test'e HİÇ görüntü
   akıtılmaz (`test_to_train`). Sonuç: no-surgical-cap 1.1k→10.6k,
   no-surgical-gloves 1.3k→15.3k, no-facial-gear 0.9k→13.6k eğitim
   örneği (10-15 kat, gerçek-ortam görüntüleriyle); test seti birebir
   SABİT (535 görüntü) → eski/yeni model karşılaştırması elmayla elma.
   Not: makalenin birleştirilmiş 31k'lık seti de açık:
   https://doi.org/10.5281/zenodo.16329852 (ileride gerekirse).

**Kalan adımlar:**

4. Eğitim bitince test setinde karşılaştır (app'teki "Metrikleri Hesapla"
   veya `evaluate.py`) — iyileşme varsa yeni `best.pt`'yi
   `MedicalPPE/best.pt`'ye kopyala ve `DEFAULT_THRESHOLDS`'ı yeni
   çıktıyla güncelle.
5. **Gerçek ortamda gözle doğrula**: test seti hâlâ ds1-ds3 kaynaklı
   (kısmen stok-fotoğraf tarzı); gerçek hedef ortamdan birkaç
   fotoğraf/videoyla davranışa bak.
6. `C:\ppe_raw` altındaki ham veri işi bitince silinebilir —
   `prepare_dataset.py` gerektiğinde yeniden indirir.
