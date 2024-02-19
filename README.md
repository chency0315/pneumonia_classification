# pneumonia_classification
This dataset is from Kaggle https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

# prepare the dataset
Download the dataset from Kaggle or use the API to call the dataset.

In this dataset, there are 4172 images for training, 1044 for validation, 624 for testing

# Generate data and data augmentation 

Using a few techniques to create more image data like sharpening, horizontal flipping, random shear, scale

the size of the image, and translate the pixels to get more training data.

```
aug = iaa.pillike.FilterSharpen() #sharpen images
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5), # 50% horizontal flip
            iaa.Affine(
                shear=(-16,16), # random shear -16 ~ +16 degree
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale x, y: 80%~120%
                translate_px={"x": (-20, 20), "y": (-20, 20)}, # Translate images by -20 to 20 pixels on x- and y-axis independently 
            ),
        ])
```


