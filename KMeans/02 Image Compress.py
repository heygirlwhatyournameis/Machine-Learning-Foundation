import numpy as np 
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Preprocessing
filename = './dolphin/image_0048.jpg'
orginal_image = np.array(Image.open(filename),dtype=np.float64) / 255

height,width,depth = orginal_image.shape
image_flattened = np.reshape(orginal_image,(width * height , depth))

# Estimator
image_sample = shuffle(image_flattened)[:1000]
estimator = KMeans(n_clusters=16,random_state=0)
estimator.fit(image_sample)

# Prediction
cluster_assignmetns = estimator.predict(image_flattened)

# Rebuild Image
image_palette = estimator.cluster_centers_

output = np.zeros((height,width,depth))

position = 0
for i in range(height):
    for j in range(width):
        output[i,j,:] = image_palette[cluster_assignmetns[position]]
        position += 1

# Display
# plt.subplot(121)
# plt.axis('off')
# plt.title('orginal')
# plt.imshow(orginal_image)
# plt.subplot(122)
# plt.axis('off')
# plt.title('compressed')
# plt.imshow(output)
# plt.show()

# Show Palette

font = FontProperties(fname='./font/msyh.ttc', size=14)  

plt.subplot(131)
plt.axis('off')
plt.title('原图',fontproperties=font)
plt.imshow(orginal_image)

plt.subplot(132)
palette = image_palette.reshape(4,4,-1)
plt.imshow(palette)
plt.axis('off')
plt.title('调色盘',fontproperties=font)

plt.subplot(133)
plt.axis('off')
plt.title('compressed')
plt.imshow(output)
plt.title('重绘结果',fontproperties=font)

plt.show()

