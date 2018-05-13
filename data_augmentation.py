# Standardize images across the dataset, mean=0, stdev=1
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from matplotlib import pyplot
from keras import backend as K
import os
K.set_image_dim_ordering('th')

def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


shift = 0.2
augmentation_list = {
	"featurewise":{"featurewise_center" : True,"featurewise_std_normalization" :True},
	"zca" : {"zca_whitening" : True},
	"rotation":{"rotation_range" : 20},
	"shifts":{"width_shift_range" : shift,"height_shift_range": shift},
	"flips":{"horizontal_flip" : True,"vertical_flip":True}
}
for features in augmentation_list:
	folder = 'images/{}/'
	datagen = ImageDataGenerator(**augmentation_list[features])
	gen_data = datagen.flow_from_directory('train_61326/',batch_size=1)
	x = ["" for i in range(len(gen_data.class_indices))]
	for k in gen_data.class_indices:
		x[gen_data.class_indices[k]] = k
		ensure_dir_exists(folder.format(k))
	output_path = folder + '{}_{}.jpg'
	for i in range(0,200):
		image,label = gen_data.next()
		print label[0]
		new_image = array_to_img(image[0], scale=True)
		new_image.save(output_path.format(x[label[0].tolist().index(1)],features,i + 1))
