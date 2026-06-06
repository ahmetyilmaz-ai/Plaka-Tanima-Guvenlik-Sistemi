# Araç Plaka Tanıma ve Güvenlik Sistemi

Bu proje, Görsel Dil Modelleri (VLM) dersi kapsamında geliştirdiğim bir bilgisayarlı görü sistemidir. Amacım, bir güvenlik noktasından geçen araçları otomatik olarak tanıyıp plaka analizi yapan ve yetkilendirme kararı veren uçtan uca bir sistem kurmaktı.

## Projenin Amacı

Sistem, modern bilgisayarlı görü ve derin öğrenme tekniklerini kullanarak aşağıdaki işlemleri gerçekleştirir:

1. **Araç Tipi Tespiti:** Görüntüdeki aracın türünü (otomobil, otobüs, kamyon, motosiklet) belirler.
2. **Plaka Tespiti ve Okuma (OCR):** Araç üzerindeki plakayı tespit eder ve çoklu stratejili bir OCR sistemiyle okur.
3. **Detaylı Araç Analizi (VLM):** Aracın rengi, markası ve durumu gibi görsel detayları analiz etmek için Florence-2 modeli kullanılır.
4. **Yetkilendirme ve Kayıt:** Okunan plakayı veritabanındaki izinli plakalar listesiyle karşılaştırarak geçişe onay veya ret kararı verir. Tüm geçiş denemeleri veritabanına kaydedilir.

## Kullanılan Teknolojiler

- **Python**
- **YOLO11** (Ultralytics) — araç tespiti
- **Florence-2** (Hugging Face Transformers) — prompt tabanlı görsel açıklama (VLM)
- **EasyOCR** (tr + en) ana motor + **Tesseract** yedek motor — plaka okuma
- **OpenCV & NumPy** — görüntü ön işleme ve veri manipülasyonu
- **SQLite** — izinli plaka ve geçiş kayıtları
- **Streamlit** — web arayüzü

## Proje Yapısı

```
VLM_Final_Projesi/
├── main.py                          # Sistemin ana çalışma dosyası (komut satırı)
├── app.py                           # Streamlit web arayüzü
├── plate_ocr.py                     # Gelişmiş çoklu-stratejili plaka OCR modülü
├── database.py                      # Veritabanı işlemleri (SQLite)
├── util.py                          # Yardımcı fonksiyonlar (format, normalize, görselleştirme)
├── yolo_train.py                    # YOLO11 araç tipi sınıflandırma eğitimi
├── requirements.txt                 # Gerekli Python kütüphaneleri
├── guvenlik_sistemi.db              # SQLite veritabanı
├── models/                          # Eğitilmiş / özel modeller
├── images/                          # Test resimleri
├── dataset/                         # Eğitim veri seti
├── outputs/                         # İşlenmiş çıktı görüntüleri
├── logs/                            # Sistem logları
└── README.md                        # Bu dosya
```

## Teknik İş Akışı

1. **Başlatma:** `main.py` veya `app.py` çalıştığında veritabanını kontrol eder, tablolar yoksa oluşturur ve örnek izinli plakaları ekler. Gelişmiş OCR sistemi (`plate_ocr.py`) önceden yüklenir.
2. **Araç Tespiti:** YOLO11 COCO modeli (`yolo11n.pt`) ile görüntüdeki araçlar tespit edilir.
3. **Plaka Bölgesi:** Özel bir plaka tespit modeli varsa onunla, yoksa aracın alt yarısı plaka bölgesi olarak alınır.
4. **OCR:** Plaka bölgesi `plate_ocr.py` modülüne gönderilir. 50+ ön işleme varyasyonu üretilir, EasyOCR (ana) ve Tesseract (yedek) ile okunur, sonuçlar oylama ile birleştirilerek en güvenilir plaka seçilir.
5. **VLM Analizi:** Florence-2 modeli `<DETAILED_CAPTION>` prompt'u ile araç hakkında bir açıklama üretir; bu açıklamadan renk, marka ve durum bilgisi çıkarılır.
6. **Karar ve Kayıt:** Plaka veritabanında sorgulanır, karara göre geçiş işlemi loglanır.
7. **Görselleştirme:** `main.py` sonuçları görüntü üzerine yazıp `outputs/` klasörüne kaydeder. `app.py` ise sonuçları web arayüzünde gösterir.

## Koddan Önemli Kısımlar

### Florence-2 ile Araç Analizi

`vlm_ile_arac_analizi` fonksiyonu, Florence-2 modelini prompt tabanlı çalıştırarak araç hakkında detaylı bir açıklama üretir. Üretilen açıklamadan aracın rengi, markası ve durumu ayıklanır.

```python
task_prompt = "<DETAILED_CAPTION>"
inputs = vlm_processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=80, num_beams=3)

generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = vlm_processor.post_process_generation(
    generated_text, task=task_prompt, image_size=image.size
)
caption = parsed.get(task_prompt, generated_text)
```

### Çoklu Stratejili Plaka OCR (Oylama)

`plate_ocr.py` içindeki `AdvancedPlateOCR` sınıfı, plaka görüntüsünü onlarca farklı ön işleme varyasyonundan geçirir ve her versiyonu OCR ile okur. Aynı plakaya gelen okumalar bir oylama sistemiyle skorlanır; en yüksek skoru alan plaka nihai sonuç olarak seçilir.

```python
# Skor: oy sayısı * 3 + ortalama güven * 2 + maksimum güven * 1
final_score = vote_count * 3 + avg_score * 2 + max_score * 1

# En yüksek skorlu plakayı seç
best_plate = max(plate_stats.items(), key=lambda x: x[1]['final_score'])
```

## Kurulum

### 1. Sanal Ortam Oluşturun (Tavsiye Edilir)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

> **Not:** Florence-2 için güncel bir `transformers` sürümü gerekir. EasyOCR, ilk çalıştırmada model dosyalarını otomatik indirir; GPU mevcutsa otomatik olarak kullanılır.

### 3. Tesseract (Yedek OCR Motoru — Opsiyonel)

Tesseract yedek motor olarak kullanılır. Sisteminizde kurulu değilse EasyOCR tek başına çalışmaya devam eder. Kurmak isterseniz Tesseract'ı işletim sisteminize göre ayrıca kurmanız gerekir.

## Kullanım

### Komut Satırı ile Kullanım

`images` klasöründeki tüm resimleri işlemek için:

```bash
python main.py
```
