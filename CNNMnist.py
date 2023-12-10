import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras

train = pd.read_csv("train.csv")
print(train.shape)
train.head

test = pd.read_csv("test.csv")
print(test.shape)
test.head()

Y_train = train["label"]

X_train = train.drop(labels = ["label"] , axis = 1)
# X train , Y'nin label kısmının çıkmış halidir.

Y_train.head()

# Data setimizdeki dağılımı görelim.

#plt.figure(figsize=(15,7))
#sns.countplot(Y_train , palette = "icefire") maalesef çıktı vermiyor
#plt.title("Number of digit classes")
Y_train.value_counts()

# plot some samples

img = X_train.iloc[0].values
# Alıp matrix haline getiriyorum.

img = img.reshape((28,28))
# 28*28 yapıyorum

plt.imshow(img , cmap = "gray")
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

img = X_train.iloc[3].values
img = img.reshape((28,28))
plt.imshow(img , cmap = "gray")
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()

# keras 2 boyutta çalışmaz. 28*28*1 şeklinde 3D vermemiz lazım. O yüzden boyutlandırmayı düzeltmemiz gerekecek.

X_train = X_train / 255.0  # normalize ediyoruz. 255 e bölmemiz yeterli.
test = test / 255.0
print("x_train shape :" , X_train.shape)
print("test shape :" , test.shape)

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape :" , X_train.shape)
print("test shape :" , test.shape) #reshape kullanarak 4 boyuta çektim matrisimi.

from tensorflow.keras.utils import to_categorical # conver to one-hot-encoding , label encoding yaptım.
Y_train = to_categorical(Y_train , num_classes = 10)

from sklearn.model_selection import train_test_split
X_train , X_val , Y_train , Y_val = train_test_split(X_train , Y_train , test_size = 0.1 , random_state = 2)

print("x_train shape" , X_train.shape)
print("x_test shape" , X_val.shape)
print("y_train shape" , Y_train.shape)
print("y_test shape" , Y_val.shape)

plt.imshow(X_train[2][:,:,0],cmap = "gray")
plt.show()

from sklearn.metrics import pair_confusion_matrix
import itertools

from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPool2D
from keras.optimizers import RMSprop , Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential() # Layerleri içinde tutuyor.

model.add(Conv2D(filters = 8 , kernel_size = (5,5) , padding = "Same" ,
                  activation = "relu" , input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16 , kernel_size = (3,3) , padding = "Same" ,
                 activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2) , strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256 , activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10 , activation = "softmax"))

# Define the optimize

optimizer = Adam(lr = 0.001 , beta_1 = 0.9 , beta_2 = 0.999)

# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy" , metrics = ["accuracy"])

epochs = 10
batch_size = 250

# data augmentation

datagen = ImageDataGenerator(
    featurewise_center = False , # set input mean to 0 over the dataset
    samplewise_center = False , # set each sample mean to 0
    featurewise_std_normalization = False , # divide inputs by std of the dataset
    samplewise_std_normalization = False , # divide each input by its std
    zca_whitening = False , # dimension reduction
    rotation_range = 0.5 , # randomly rotate images in the range 15 degrees
    zoom_range = 0.5 , # randomly zoom image %15
    width_shift_range = 0.5 , # randomly shift images horizontally %15
    height_shift_range = 0.5 , # randomly shift images vertically %15
    horizontal_flip = False , # randomly flip images
    vertical_flip = False) # randomly flip images

datagen.fit(X_train)

# fit the model

history = model.fit_generator(datagen.flow(X_train , Y_train , batch_size = batch_size),
                              epochs = epochs , validation_data = (X_val , Y_val) , steps_per_epoch = X_train.shape [0] // batch_size)

# validation loss

plt.plot(history.history["val_loss"] , color = "r" , label = "validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix

import seaborn as sns
from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred , axis = 1)

Y_true = np.argmax(Y_val , axis = 1)

confusion_mtx = confusion_matrix(Y_true , Y_pred_classes)

f , ax = plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx , annot = True , linewidths = 0.01 , cmap = "Greens" , linecolor = "gray" , fmt =".1f" , ax = ax)
plt.xlabel("Predict Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

