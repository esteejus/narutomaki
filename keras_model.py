import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate,Input 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


#Seeding
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

#Hyperparameters
batch_size = 8
lr = 1e-4
epochs = 100
height = 256
width  = 256
 
#Path
dataset_path = os.path.join("./","data")

files_dir  = os.path.join("files","data")
model_file = os.path.join(files_dir, "unet-data.h5")
log_file   = os.path.join(files_dir, "log-data.csv")

#Create folder
def create_dir(path):
     if not os.path.exists(path):
         os.makedirs(path)

create_dir(files_dir)

#Building UNET
#=====================

#Conv block
def conv_block(inputs,num_filters):
    x = Conv2D(num_filters, 2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

#Encoder Block
def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

#Decoder Block
def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
    x = Concatenate()([x,skip])
    x = conv_block(x, num_filters)
    return x

#UNET
def build_unet(input_shape):
    inputs = Input(input_shape)

    #Encoder
    s1, p1 = encoder_block(inputs,64)
    s2, p2 = encoder_block(p1,128)
    s3, p3 = encoder_block(p2,256)
    s4, p4 = encoder_block(p3,512)

    #Bride
    b1 = conv_block(p4, 1024)

    #Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 63)

    outputs = Conv2D(1,1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="UNET")
    return model

#Loading the training and validation data
def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*")))
    train_y = sorted(glob(os.path.join(path, "train", "masks", "*")))

    valid_x = sorted(glob(os.path.join(path, "valid", "images", "*")))
    valid_y = sorted(glob(os.path.join(path, "valid", "masks", "*")))

    return (train_x, train_y), (valid_x, valid_y)

#Read images
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

#Reading Mask
def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x,y):
    def _parse(x,y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x,y], [tf.float64,tf.float64])
    x.set_shape([height,width,1])
    y.set_shape([height,width,1])
    return x, y

def tf_dataset(x, y , batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


#Training
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)
print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

for x, y in train_dataset:
    print(x.shape,y.shape)

input_shape = (height, width, 1)
model = build_unet(input_shape)

model.summary()

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])
callback = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2),
    CSVLogger(log_file),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=callback
    )






'''
data_dir = './data/images/'
val_dir = './data/masks/'

batch_size = 10


seed = 909 # (IMPORTANT) to transform image and corresponding mask with same augmentation parameter.
image_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1)
#                                   preprocessing_function = image_preprocessing) # custom fuction for each image you can use resnet one too.
mask_datagen = ImageDataGenerator(width_shift_range=0.1,
                                  height_shift_range=0.1)
 #                                 preprocessing_function = mask_preprocessing)  # to make mask as feedable formate (256,256,1)

image_generator =image_datagen.flow_from_directory(data_dir,
                                                   class_mode=None, seed=seed)

mask_generator = mask_datagen.flow_from_directory(val_dir,
                                                  class_mode=None, seed=seed)

train_generator = zip(image_generator, mask_generator)



'''
