import os
os.environ['NUMEXPR_MAX_THREADS'] = '4'
import albumentations as a
import cv2
import numpy as np


def apply_random_rotate(image):
    rotate = a.Rotate(limit=45)
    return rotate(image=image)['image']


def apply_horizontal_flip(image):
    flip = a.HorizontalFlip()
    return flip(image=image)['image']


def apply_random_brightness_contrast(image):
    brightness_contrast = a.RandomBrightnessContrast()
    return brightness_contrast(image=image)['image']


def apply_random_blur(image):
    blur = a.Blur(blur_limit=(3, 7), p=0.5)
    return blur(image=image)['image']


def apply_random_gamma(image):
    slika = a.RandomGamma(gamma_limit=(80, 120))
    return slika(image=image)['image']


def apply_random_fog(image):
    slika = a.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5)
    return slika(image=image)['image']


def apply_random_rain(image):
    slika = a.RandomRain(drop_width=1, blur_value=7)
    return slika(image=image)['image']


def apply_random_snow(image):
    slika = a.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5)
    return slika(image=image)['image']


def apply_motion_blur(image):
    kernel = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]) / 7.0

    motion_blurred_image = cv2.filter2D(image, -1, kernel)
    return motion_blurred_image


def apply_random_shadow(image):
    slika = a.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5)
    return slika(image=image)['image']


def transform(image):
    if np.random.rand() < 0.2:
        image = apply_random_fog(image)
    if np.random.rand() < 0.3:
        image = apply_horizontal_flip(image)
    if np.random.rand() < 0.2:
        image = apply_random_brightness_contrast(image)
    if np.random.rand() < 0.2:
        image = apply_random_rotate(image)
    if np.random.rand() < 0.2:
        image = apply_random_gamma(image)
    if np.random.rand() < 0.5:
        image = apply_motion_blur(image)
    if np.random.rand() < 0.3:
        image = apply_random_shadow(image)
    if np.random.rand() < 0.2:
        image = apply_random_rain(image)
    if np.random.rand() < 0.2:
        image = apply_random_snow(image)
    if np.random.rand() < 0.2:
        image = apply_random_blur(image)

    return {'image': image}


def main():
    input_dir = "Imgs"
    output_dir = "augmented_images"
    os.makedirs(output_dir, exist_ok=True)

    for image_filename in os.listdir(input_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output_image_dir = os.path.join(output_dir, os.path.splitext(image_filename)[0])
            os.makedirs(output_image_dir, exist_ok=True)

            for i in range(100):
                transformed = transform(image)['image']
                output_image_path = os.path.join(output_image_dir, f"augmented_image_{i}.png")
                cv2.imwrite(output_image_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))

            print(f"Generated 100 augmented images for {image_filename} in '{output_image_dir}'.")


if __name__ == '__main__':
    main()
