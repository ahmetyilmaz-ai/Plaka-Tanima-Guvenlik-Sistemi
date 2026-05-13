"""
Araç Plaka Tanıma ve Güvenlik Sistemi
Proje Ana Dosyası - Görüntüleri işler, araçları tespit eder, plakaları okur ve yetkilendirme yapar
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Gelişmiş plaka OCR modülü
from plate_ocr import read_plate, get_ocr_instance

# UTF-8 encoding için Windows düzeltmesi
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent
# OpenMP çakışması için Windows düzeltmesi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# GPU kullanımı için ayar
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'sistem.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== VERİTABANI FONKSİYONLARI ====================

import database as db


# ==================== PLAKA OKUMA FONKSİYONLARI ====================

def plaka_oku_coklu_deneme(plate_img):
    """
    Plaka görüntüsünü gelişmiş çoklu-stratejili OCR ile okur
    MMOCR + EasyOCR + Tesseract kombinasyonu kullanır
    """
    try:
        result = read_plate(plate_img, use_gpu=USE_GPU)
        return result
    except Exception as e:
        logger.error(f"Plaka okuma hatası: {e}")
        return "OKUNAMADI"


# ==================== VLM FONKSİYONLARI ====================

def vlm_ile_arac_analizi(image_path, arac_tipi):
    """
    BLIP modeli kullanarak araç hakkında görsel analiz yapar
    Rengi, markası ve durumu hakkında bilgi döndürür - GPU destekli
    """
    global vlm_processor, vlm_model

    try:
        if 'vlm_processor' not in globals():
            logger.info(f"VLM modeli yükleniyor (GPU: {USE_GPU})...")
            vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            vlm_model.to(DEVICE)
            vlm_model.eval()
            logger.info(f"VLM modeli yüklendi (device: {DEVICE})")

        image = Image.open(image_path).convert('RGB')
        inputs = vlm_processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_length=50)

        caption = vlm_processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"VLM Caption: {caption}")

        # Caption'dan araç bilgilerini çıkar
        yorum = caption

        # Renk tespiti
        renkler = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'silver']
        renk = "UNKNOWN"
        for r in renkler:
            if r.lower() in caption.lower():
                renk = r.upper()
                break

        # Marka tespiti (basit)
        markalar = ['toyota', 'bmw', 'mercedes', 'audi', 'volkswagen', 'fiat', 'renault',
                    'honda', 'hyundai', 'ford', 'peugeot', 'citroen']
        marka = "UNKNOWN"
        for m in markalar:
            if m in caption.lower():
                marka = m.upper()
                break

        # Durum tespiti
        durum = "PARKED" if 'parked' in caption.lower() else "MOVING"

        yorum = f"{renk} {marka} {arac_tipi} ({durum})"
        logger.info(f"VLM Comment: {yorum}")

        return yorum

    except Exception as e:
        logger.error(f"VLM hatası: {e}")
        return "VLM ANALYSIS FAILED"


# ==================== GÖRSELLEŞTİRME ====================

def sonucları_gorsellestir(image_path, arac_tipi, plaka, vlm_yorumu, karar):
    """
    Analiz sonuçlarını görüntü üzerine yazdırıp gösterir
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Görüntü okunamadı: {image_path}")
        return

    h, w = img.shape[:2]

    # Görüntü boyutuna göre yazı boyutlarını ayarla
    scale = max(1.0, min(w, h) / 1000.0)  # 1000px'de scale=1

    # Renkler
    yesil = (0, 200, 0)
    kirmizi = (0, 0, 220)
    beyaz = (240, 240, 240)
    sari = (0, 200, 200)
    mavi = (200, 100, 0)
    siyah = (10, 10, 10)

    renk = yesil if karar == "ALLOWED" else kirmizi

    # Bilgi paneli - görüntü boyutuna göre ayarla
    panel_h = int(100 * scale)  # Increased panel height
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Çizgi kalınlığı ve font boyutları
    line_thick = max(1, int(2 * scale))
    font_scale_large = 0.7 * scale
    font_scale_medium = 0.6 * scale
    font_scale_small = 0.5 * scale

    y_offset = int(30 * scale)
    line_spacing = int(32 * scale)

    # Satır 1: Karar (en büyük, renkli arka plan)
    karar_text = f"STATUS: {karar}"
    (tw, th), _ = cv2.getTextSize(karar_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, line_thick + 1)
    cv2.rectangle(img, (10, y_offset - th - 10), (20 + tw, y_offset + 5), renk, -1)
    cv2.putText(img, karar_text, (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, siyah, line_thick + 1)

    # Satır 2: Araç tipi ve Plaka (yan yana)
    y_offset += line_spacing
    tip_text = f"{arac_tipi}"
    plaka_text = f"PLATE: {plaka}"

    (tw1, th1), _ = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, line_thick)
    (tw2, th2), _ = cv2.getTextSize(plaka_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, line_thick)

    x_pos = 15
    cv2.putText(img, tip_text, (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, beyaz, line_thick)

    x_pos += tw1 + int(50 * scale)
    cv2.putText(img, plaka_text, (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, sari, line_thick)

    # Satır 3: VLM yorumu
    y_offset += line_spacing
    cv2.putText(img, f"VLM: {vlm_yorumu}", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, mavi, max(1, line_thick - 1))

    # Sağ alt köşe zaman damgası (küçük)
    zaman = datetime.now().strftime('%H:%M:%S')
    cv2.putText(img, zaman, (w - 100, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Görüntüyü göster (daha küçük pencere)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Kaydet
    output_path = PROJECT_ROOT / "outputs" / f"sonuc_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), img)
    logger.info(f"Sonuç kaydedildi: {output_path}")


# ==================== ANA İŞ AKIŞI ====================

def main():
    """
    Ana iş akışı - tüm sistemi yönetir
    """
    logger.info("=" * 60)
    logger.info("ARAÇ PLAKA TANİMA VE GÜVENLİK SİSTEMİ")
    logger.info("=" * 60)

    # 1. Veritabanını hazırla
    db.init_database()

    # 1.5. Gelişmiş OCR sistemini önceden yükle
    logger.info("Gelişmiş OCR sistemi başlatılıyor...")
    get_ocr_instance(use_gpu=USE_GPU)
    logger.info("OCR sistemi hazır")

    # 2. Modelleri yükle
    logger.info("Modeller yükleniyor...")

    # COCO modeli (araç tespiti için)
    coco_model = YOLO('yolov8n.pt')
    logger.info("COCO modeli yüklendi")

    # Plaka tespit modeli (varsa)
    new_plaka_model_path = PROJECT_ROOT / "models" / "license_plate_detector_v2.pt"
    old_plaka_model_path = PROJECT_ROOT / "models" / "license_plate_detector.pt"
    plaka_model = None

    if new_plaka_model_path.exists():
        plaka_model = YOLO(str(new_plaka_model_path))
        logger.info("Yeni plaka tespit modeli yüklendi")
    elif old_plaka_model_path.exists():
        plaka_model = YOLO(str(old_plaka_model_path))
        logger.info("Eski plaka tespit modeli yüklendi")
    else:
        logger.warning("Plaka tespit modeli bulunamadı, COCO modeli kullanılacak")

    # 3. Görüntüleri işle
    images_dir = PROJECT_ROOT / "images"

    if not images_dir.exists():
        logger.error(f"Görüntü klasörü bulunamadı: {images_dir}")
        logger.info("Lütfen 'images' klasörüne araç resimleri ekleyin")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        logger.error("Görüntü bulunamadı")
        return

    logger.info(f"{len(image_files)} görüntü işlenecek")

    # Her görüntüyü işle
    for image_path in image_files:
        logger.info("-" * 40)
        logger.info(f"İşleniyor: {image_path.name}")

        # 1. Araç tespiti
        results = coco_model(image_path, verbose=False)
        arac_bulundu = False
        arac_tipi = "BİLİNMEYEN"
        arac_bbox = None

        COCO_CLASSES = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }

        VEHICLE_NAMES = {
            'car': 'CAR',
            'motorcycle': 'MOTORCYCLE',
            'bus': 'BUS',
            'truck': 'TRUCK'
        }

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in COCO_CLASSES:
                    arac_tipi = COCO_CLASSES[cls_id]
                    arac_bbox = box.xyxy[0].tolist()
                    arac_bulundu = True
                    break
            if arac_bulundu:
                break

        if not arac_bulundu:
            logger.warning("Araç tespit edilemedi")
            continue

        arac_tipi_en = VEHICLE_NAMES.get(arac_tipi, arac_tipi.upper())
        logger.info(f"Araç tespit edildi: {arac_tipi_en}")

        # Araç görüntüsünü kes
        img = cv2.imread(str(image_path))
        x1, y1, x2, y2 = map(int, arac_bbox)
        arac_img = img[y1:y2, x1:x2]

        # 2. Plaka tespiti ve okuma
        plaka_bulundu = False
        plaka_img = None

        if plaka_model:
            # Özel plaka modeli ile tespit dene
            plaka_results = plaka_model(arac_img, verbose=False)
            for pr in plaka_results:
                for pbox in pr.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                    plaka_img = arac_img[py1:py2, px1:px2]
                    plaka_bulundu = True
                    break
                if plaka_bulundu:
                    break

        # Plaka bulunamadıysa araç görüntüsünün alt yarısını kullan
        if not plaka_bulundu:
            h, w = arac_img.shape[:2]
            plaka_img = arac_img[int(h/2):h, :]
            logger.info("Plaka bölgesi tespit edilemedi, alt bölüm kullanılıyor")

        # Plakayı oku
        plaka_text = plaka_oku_coklu_deneme(plaka_img)

        # 3. VLM ile araç analizi
        vlm_yorumu = vlm_ile_arac_analizi(image_path, arac_tipi_en)

        # 4. Yetkilendirme kontrolü
        if arac_tipi_en != 'CAR':
            karar = "DENIED"
            sebep = "Vehicle type not authorized"
        elif plaka_text == "OKUNAMADI":
            karar = "DENIED"
            sebep = "Plate not readable"
        elif db.plaka_izinli_mi(plaka_text):
            karar = "ALLOWED"
            sebep = "Authorized plate"
        else:
            karar = "DENIED"
            sebep = "Unauthorized plate"

        logger.info(f"Karar: {karar} ({sebep})")

        # 5. Log kaydı
        db.log_kaydet(plaka_text, arac_tipi_en, vlm_yorumu, karar)

        # 6. Görselleştirme
        sonucları_gorsellestir(str(image_path), arac_tipi_en, plaka_text, vlm_yorumu, karar)

    logger.info("=" * 60)
    logger.info("İŞLEM TAMAMLANDI")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
