import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST veri setini yükle
mnist = tf.keras.datasets.mnist

# Veri setini eğitim ve test olarak böl
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Verileri normalize et (0-1 arasına ölçekle)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Bir yapay sinir ağı modeli oluştur
model = tf.keras.models.Sequential([
    # Giriş katmanı olarak düzleştirici katman kullan (28x28 piksel görüntüleri 784 boyutlu vektörlere dönüştür)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Gizli katman olarak 128 nöronlu yoğun katman kullan (ReLU aktivasyon fonksiyonu ile)
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Damla katmanı ekle (%20 damla oranı ile)
    tf.keras.layers.Dropout(0.2),
    
    # Gizli katman olarak 64 nöronlu yoğun katman kullan (ReLU aktivasyon fonksiyonu ile)
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Damla katmanı ekle (%10 damla oranı ile)
    tf.keras.layers.Dropout(0.1),
    
    # Çıkış katmanı olarak 10 nöronlu yoğun katman kullan (Softmax aktivasyon fonksiyonu ile, 10 sınıf için olasılık dağılımı üret)
    tf.keras.layers.Dense(10, activation='softmax')
])

# Modeli derle (kayıp fonksiyonu olarak çapraz entropi, optimizasyon algoritması olarak Adam, başarı metriği olarak doğruluk kullan)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelin eğitim tarihçesini al
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Modelin test verisi üzerindeki performansını değerlendir (kayıp ve doğruluk değerlerini yazdır)
model.evaluate(x_test, y_test, verbose=2)

# Eğitim ve test verileri üzerindeki kayıp değerlerini al
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# Eğitim ve test verileri üzerindeki doğruluk değerlerini al
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

# Dönem sayısını al
epochs = range(1, len(train_loss) + 1)

# Kayıp eğrilerini çiz
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Eğitim kaybı')
plt.plot(epochs, test_loss, 'r', label='Test kaybı')
plt.title('Eğitim ve Test Kaybı')
plt.xlabel('Dönem')
plt.ylabel('Kayıp')
plt.legend()

# Doğruluk eğrilerini çiz
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Eğitim doğruluğu')
plt.plot(epochs, test_acc, 'r', label='Test doğruluğu')
plt.title('Eğitim ve Test Doğruluğu')
plt.xlabel('Dönem')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()
