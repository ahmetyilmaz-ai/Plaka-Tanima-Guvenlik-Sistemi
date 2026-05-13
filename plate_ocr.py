"""
Gelişmiş Plaka OCR Modülü
EasyOCR tabanlı çoklu-stratejili plaka okuma sistemi
Güçlendirilmiş ön işleme adımları ile yüksek doğruluk
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class AdvancedPlateOCR:
    """
    Gelişmiş plaka okuma sınıfı
    EasyOCR + 50+ ön işleme stratejisi
    """

    def __init__(self, use_gpu: bool = True):
        """
        OCR sistemini başlatır

        Args:
            use_gpu: GPU kullanım bayrağı
        """
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        self.tesseract_available = False

        # Başlat
        self._init_easyocr()
        self._init_tesseract()

    def _init_easyocr(self):
        """EasyOCR modelini başlatır - Ana motor"""
        try:
            import easyocr
            logger.info(f"EasyOCR yükleniyor (GPU: {self.use_gpu})...")
            self.easyocr_reader = easyocr.Reader(['en', 'tr'], gpu=self.use_gpu)
            logger.info("EasyOCR başarıyla yüklendi (Ana OCR motoru)")
        except ImportError:
            logger.warning("EasyOCR kurulu değil")
        except Exception as e:
            logger.warning(f"EasyOCR başlatılamadı: {e}")

    def _init_tesseract(self):
        """Tesseract'ı kontrol eder - Yedek motor"""
        try:
            import pytesseract
            import shutil
            self.tesseract_available = shutil.which("tesseract") is not None
            if self.tesseract_available:
                logger.info("Tesseract mevcut (yedek motor)")
        except ImportError:
            logger.debug("pytesseract kurulu değil")

    def preprocess_image(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Plaka görüntüsü için çoklu ön işleme stratejileri
        50+ farklı varyasyon üretir

        Args:
            img: Giriş görüntüsü

        Returns:
            İşlenmiş görüntü listesi
        """
        processed_images = []

        h, w = img.shape[:2]

        # Görüntü çok küçükse büyüt
        if h < 50 or w < 100:
            scale = max(3.0, 200 / max(h, w))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        # Gri tonlamaya çevir
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # ========================================
        # GRUP 1: Temel görüntüler
        # ========================================
        processed_images.append(gray)
        processed_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else gray)

        # ========================================
        # GRUP 2: Unsharp Masking - Keskinleştirme
        # ========================================
        # Farklı sigma ve alpha kombinasyonları
        for sigma in [1.5, 2.0, 2.5, 3.0, 4.0]:
            gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            for alpha in [1.5, 1.8, 2.0, 2.2]:
                try:
                    unsharp = cv2.addWeighted(gray, alpha, gaussian, -(alpha-1), 0)
                    processed_images.append(unsharp)
                except:
                    pass

        # ========================================
        # GRUP 3: CLAHE - Kontrast Artırma
        # ========================================
        # Farklı parametrelerle CLAHE
        for clip_limit in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            for grid_size in [(2, 2), (4, 4), (8, 8), (16, 16)]:
                try:
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
                    clahe_img = clahe.apply(gray)
                    processed_images.append(clahe_img)
                except:
                    pass

        # ========================================
        # GRUP 4: Unsharp + CLAHE Kombinasyonları
        # ========================================
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        for sigma in [2.0, 2.5, 3.0]:
            try:
                gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
                unsharp = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
                clahe_unsharp = clahe.apply(unsharp)
                processed_images.append(clahe_unsharp)
            except:
                pass

        # ========================================
        # GRUP 5: Histogram Equalization
        # ========================================
        try:
            eq_hist = cv2.equalizeHist(gray)
            processed_images.append(eq_hist)

            # CLAHE + Histogram Eq
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_images.append(clahe.apply(eq_hist))
        except:
            pass

        # ========================================
        # GRUP 6: Adaptive Thresholding
        # ========================================
        base_images = [gray]
        # Unsharp örneklerini ekle
        for sigma in [2.0, 3.0]:
            try:
                gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
                unsharp = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
                base_images.append(unsharp)
            except:
                pass

        # CLAHE örneklerini ekle
        for clip_limit in [2.0, 3.0]:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                base_images.append(clahe.apply(gray))
            except:
                pass

        # Her base görüntü için adaptive threshold uygula
        for base_img in base_images[:5]:  # İlk 5 base görüntü
            # Gauss adaptive threshold - farklı parametrelerle
            for block_size in [9, 11, 13, 15, 17]:
                for c_val in [1, 2, 3, 4]:
                    try:
                        adaptive_gauss = cv2.adaptiveThreshold(
                            base_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, block_size, c_val
                        )
                        processed_images.append(adaptive_gauss)
                        processed_images.append(cv2.bitwise_not(adaptive_gauss))
                    except:
                        pass

            # Mean adaptive threshold
            for block_size in [9, 11, 13]:
                for c_val in [2, 3]:
                    try:
                        adaptive_mean = cv2.adaptiveThreshold(
                            base_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, block_size, c_val
                        )
                        processed_images.append(adaptive_mean)
                        processed_images.append(cv2.bitwise_not(adaptive_mean))
                    except:
                        pass

        # ========================================
        # GRUP 7: Morfolojik İşlemler
        # ========================================
        morph_bases = [gray, base_images[1] if len(base_images) > 1 else gray]

        for kernel_size in [(2, 2), (3, 3), (2, 3), (3, 2)]:
            kernel = np.ones(kernel_size, np.uint8)
            for base_img in morph_bases:
                try:
                    # Dilation + Erosion
                    dilated = cv2.dilate(base_img, kernel, iterations=1)
                    eroded = cv2.erode(dilated, kernel, iterations=1)
                    processed_images.append(eroded)

                    # Açma
                    opened = cv2.morphologyEx(base_img, cv2.MORPH_OPEN, kernel)
                    processed_images.append(opened)

                    # Kapatma
                    closed = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel)
                    processed_images.append(closed)

                    # Gradient
                    gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
                    processed_images.append(gradient)

                    # Top-hat
                    tophat = cv2.morphologyEx(base_img, cv2.MORPH_TOPHAT, kernel)
                    processed_images.append(tophat)

                    # Black-hat
                    blackhat = cv2.morphologyEx(base_img, cv2.MORPH_BLACKHAT, kernel)
                    processed_images.append(blackhat)
                except:
                    pass

        # ========================================
        # GRUP 8: Otsu Thresholding
        # ========================================
        otsu_bases = [gray, base_images[1] if len(base_images) > 1 else gray, base_images[2] if len(base_images) > 2 else gray]
        for base_img in otsu_bases[:3]:
            try:
                _, otsu = cv2.threshold(base_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(otsu)
                processed_images.append(cv2.bitwise_not(otsu))
            except:
                pass

        # ========================================
        # GRUP 9: Binary Thresholding (farklı değerlerle)
        # ========================================
        for threshold in [50, 80, 100, 120, 140, 160]:
            try:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                processed_images.append(binary)
                processed_images.append(cv2.bitwise_not(binary))
            except:
                pass

        # ========================================
        # GRUP 10: Gürültü Giderme + Sharpening
        # ========================================
        try:
            # Median filter
            median = cv2.medianBlur(gray, 3)
            processed_images.append(median)

            # Bilateral filter
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            processed_images.append(bilateral)

            # Non-local means denoising
            nlm = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            processed_images.append(nlm)
        except:
            pass

        # ========================================
        # GRUP 11: Kenar Algılama + Kombinasyonlar
        # ========================================
        try:
            # Canny edge
            edges = cv2.Canny(gray, 50, 150)
            processed_images.append(edges)

            # Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)
            processed_images.append(sobelx)
            processed_images.append(sobely)

            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_8U)
            processed_images.append(laplacian)
        except:
            pass

        # Görüntü sayısını logla
        logger.debug(f"Toplam {len(processed_images)} işlenmiş görüntü üretildi")

        return processed_images

    def format_plate_text(self, text: str) -> Optional[str]:
        """
        Plaka metnini doğrular ve formatlar

        Args:
            text: Okunan metin

        Returns:
            Formatlanmış plaka veya None
        """
        if not text:
            return None

        # Alfanumerik karakterleri koru
        clean_text = ''.join(filter(str.isalnum, text)).upper()

        # Karakter düzeltmeleri (görüntü benzerliği)
        corrections = {
            'O': '0', 'Q': '0',
            'I': '1', 'L': '1', '|': '1',
            'S': '5', '$': '5',
            'Z': '2',
            'B': '8',
            'G': '6',
            'T': '7',
            'D': '0',
            'A': '4',
            'H': '4'
        }

        for wrong, correct in corrections.items():
            clean_text = clean_text.replace(wrong, correct)

        # Türk plaka formatı kontrolü (esnek)
        # En az 4, en fazla 10 karakter
        if 4 <= len(clean_text) <= 10:
            return clean_text

        return None

    def read_with_easyocr(self, img: np.ndarray) -> List[Tuple[str, float]]:
        """
        EasyOCR ile plaka okuma - Ana motor

        Args:
            img: Giriş görüntüsü

        Returns:
            (plaka_metni, güven_skoru) listesi
        """
        results = []

        if self.easyocr_reader is None:
            return results

        try:
            # Farklı parametre kombinasyonları ile dene
            param_sets = [
                # Standart
                {
                    'allowlist': '0123456789ABCDEFGHJKLMNPRSTUVWXYZ',
                    'detail': 1,
                    'paragraph': False,
                    'width_ths': 0.8,
                    'height_ths': 0.8,
                    'decoder': 'beamsearch',
                    'contrast_ths': 0.3,
                    'adjust_contrast': 0.7
                },
                # Daha agresif
                {
                    'allowlist': '0123456789ABCDEFGHJKLMNPRSTUVWXYZ',
                    'detail': 1,
                    'paragraph': False,
                    'width_ths': 1.0,
                    'height_ths': 1.0,
                    'decoder': 'beamsearch',
                    'contrast_ths': 0.2,
                    'adjust_contrast': 0.5
                },
                # Daha toleranslı
                {
                    'allowlist': '0123456789ABCDEFGHJKLMNPRSTUVWXYZ',
                    'detail': 1,
                    'paragraph': False,
                    'width_ths': 0.7,
                    'height_ths': 0.7,
                    'decoder': 'beamsearch',
                    'contrast_ths': 0.4,
                    'adjust_contrast': 0.8
                }
            ]

            for params in param_sets:
                try:
                    ocr_results = self.easyocr_reader.readtext(img, **params)

                    for bbox, text, prob in ocr_results:
                        if prob > 0.25:  # Daha düşük threshold
                            formatted = self.format_plate_text(text)
                            if formatted:
                                # Aynı plaka zaten var mı kontrol et
                                exists = False
                                for existing in results:
                                    if existing[0] == formatted:
                                        exists = True
                                        break
                                if not exists:
                                    results.append((formatted, prob))
                                    logger.debug(f"EasyOCR: {formatted} ({prob:.2f})")
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"EasyOCR okuma hatası: {e}")

        return results

    def read_with_tesseract(self, img: np.ndarray) -> List[Tuple[str, float]]:
        """
        Tesseract ile plaka okuma - Yedek motor

        Args:
            img: Giriş görüntüsü

        Returns:
            (plaka_metni, güven_skoru) listesi
        """
        results = []

        if not self.tesseract_available:
            return results

        try:
            import pytesseract

            # Farklı PSM modları dene
            psm_modes = [6, 7, 11, 13, 3, 8]

            for psm in psm_modes:
                try:
                    custom_config = f'--oem 3 --psm {psm} -l eng'
                    text = pytesseract.image_to_string(img, config=custom_config)

                    if text.strip():
                        formatted = self.format_plate_text(text)
                        if formatted:
                            results.append((formatted, 0.6))  # Tesseract için orta güven
                            logger.debug(f"Tesseract (PSM{psm}): {formatted}")
                            break  # İlk başarılı okuma yeterli

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Tesseract okuma hatası: {e}")

        return results

    def read_plate(self, plate_img: np.ndarray) -> str:
        """
        Plaka görüntüsünü okuma - Ana fonksiyon

        Tüm ön işleme stratejilerini ve OCR motorlarını dener,
        en iyi sonucu oylama ile belirler.

        Args:
            plate_img: Plaka görüntüsü

        Returns:
            Okunan plaka metni veya "OKUNAMADI"
        """
        all_candidates = []

        # Ön işleme adımları
        processed_images = self.preprocess_image(plate_img)

        logger.info(f"{len(processed_images)} ön işlenmiş görüntüden plaka okunuyor...")

        # Her görüntü için tüm OCR motorlarını dene
        for i, img in enumerate(processed_images):
            # Her 25 görüntüde bir log (çok log olmasın)
            if i % 25 == 0:
                logger.debug(f"İşlenmiş görüntü {i+1}/{len(processed_images)} okunuyor...")

            # EasyOCR (ana motor)
            if self.easyocr_reader is not None:
                easyocr_results = self.read_with_easyocr(img)
                all_candidates.extend(easyocr_results)

            # Tesseract (yedek motor - sadece bazı görüntülerde)
            if self.tesseract_available and i % 5 == 0:
                tesseract_results = self.read_with_tesseract(img)
                all_candidates.extend([(r[0], r[1] * 0.8) for r in tesseract_results])

        # Oylama sistemi
        if all_candidates:
            # Aynı plakalar için oyları topla
            plate_votes = defaultdict(list)
            for plate, score in all_candidates:
                plate_votes[plate].append(score)

            # Her plakanın istatistiklerini hesapla
            plate_stats = {}
            for plate, votes in plate_votes.items():
                vote_count = len(votes)
                avg_score = sum(votes) / vote_count
                max_score = max(votes)
                sum_score = sum(votes)

                # Skor: oy sayısı * 3 + ortalama * 2 + max * 1
                final_score = vote_count * 3 + avg_score * 2 + max_score * 1
                plate_stats[plate] = {
                    'vote_count': vote_count,
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'final_score': final_score
                }

            # En yüksek final_score'a sahip plakayı seç
            best_plate = max(plate_stats.items(), key=lambda x: x[1]['final_score'])
            stats = best_plate[1]

            logger.info(f"Plaka okundu: {best_plate[0]} (oy: {stats['vote_count']}, "
                       f"ortalama: {stats['avg_score']:.2f}, max: {stats['max_score']:.2f})")
            return best_plate[0]

        logger.warning("Plaka okunamadı")
        return "OKUNAMADI"


# Global singleton instance
_ocr_instance = None


def get_ocr_instance(use_gpu: bool = True) -> AdvancedPlateOCR:
    """
    OCR singleton örneğini döndürür

    Args:
        use_gpu: GPU kullanım bayrağı

    Returns:
        AdvancedPlateOCR örneği
    """
    global _ocr_instance

    if _ocr_instance is None:
        _ocr_instance = AdvancedPlateOCR(use_gpu=use_gpu)

    return _ocr_instance


def read_plate(plate_img: np.ndarray, use_gpu: bool = True) -> str:
    """
    Kolay kullanım için wrapper fonksiyon

    Args:
        plate_img: Plaka görüntüsü
        use_gpu: GPU kullanım bayrağı

    Returns:
        Okunan plaka metni veya "OKUNAMADI"
    """
    ocr = get_ocr_instance(use_gpu=use_gpu)
    return ocr.read_plate(plate_img)


if __name__ == "__main__":
    # Test için
    import sys

    # Log config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test görüntüsü varsa dene
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)

        if img is not None:
            result = read_plate(img)
            print(f"Sonuç: {result}")
        else:
            print(f"Görüntü okunamadı: {img_path}")
    else:
        print("Kullanım: python plate_ocr.py <görüntü_yolu>")
