# Import Libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Read the image
img = cv2.imread("elephant.jpg")

# Flatten each channel of the image
all_pixels = img.reshape((img.shape[1]*img.shape[0],img.shape[2]))

#Number of colors to be segmented into
dominant_colors = 4

# Use KMeans function from Scikit-Learn
km = KMeans(n_clusters=dominant_colors)

# Train model
km.fit(all_pixels)

# Cluster Centers
centers = np.array(km.cluster_centers_, dtype='uint8')

# Transform color list in numpy array
colors = []
for color in centers:
    colors.append(np.array(color))

# Segment our original image
new_img = np.zeros((img.shape[1]*img.shape[0],img.shape[2]),dtype='uint8')
for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]
new_img = new_img.reshape((img.shape))

# Save Segmented image
cv2.imwrite("Segmented_Image.jpg", new_img)