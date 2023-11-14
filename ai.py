# TensorFlow kütüphanesini içe aktar
import tensorflow as tf

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
  # Çıkış katmanı olarak 10 nöronlu yoğun katman kullan (Softmax aktivasyon fonksiyonu ile, 10 sınıf için olasılık dağılımı üret)
  tf.keras.layers.Dense(10, activation='softmax')
])

# Modeli derle (kayıp fonksiyonu olarak çapraz entropi, optimizasyon algoritması olarak Adam, başarı metriği olarak doğruluk kullan)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Modeli eğit (eğitim verisi, dönem sayısı, grup boyutu ve doğrulama verisi belirle)
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Modelin test verisi üzerindeki performansını değerlendir (kayıp ve doğruluk değerlerini yazdır)
model.evaluate(x_test, y_test, verbose=2)
