from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from PIL import Image

import json


# Load image into RAM 
image = mpimg.imread('../cat.png')
pil_image = Image.open('../cat.png')

# Reshape image into matrix 
w, h, d = tuple(image.shape)
pixel = np.reshape(image, (w * h, d))


# Run a KMeans clustering to determine color pallete
n_colors = 4
kmeans_model = KMeans(n_clusters=n_colors, random_state=42, verbose=1).fit(pixel)

# Create array of f32's of the color pallete
color_palette = [np.float32(kmeans_model.cluster_centers_)]
print(color_palette)

with open("../color_palette.json", "w") as outfile:
    json.dump(color_palette[0].tolist(), outfile)


# For debug purposes, display the color pallete
# plt.imshow(color_pallete)
# plt.show()


