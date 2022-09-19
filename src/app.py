import repository.ImageRepository as repository
import ultrasound_detection.ContourDetector as detector
import ultrasound_detection.CropService as crop
import fetal_detection.FetalDetector as fetal

path = input('Path: ')
image = repository.read_image(path)
contour = detector.detect_ultrasound_image_contour(image)
ultrasound_image = crop.crop_image_by_contour(image, contour)
img = fetal.detect_fetal_image(ultrasound_image)
repository.write_image(img, path)
