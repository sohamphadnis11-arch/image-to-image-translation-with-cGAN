# -*- coding: utf-8 -*-
"""
Pix2Pix Image-to-Image Translation (Re–Structured Version)
Functional Output = SAME | Code Structure = DIFFERENT
"""

# ======================================================
# 📦 1. IMPORTS
# ======================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os, tarfile

print("🔥 TensorFlow Loaded — Version:", tf.__version__)

# ======================================================
# 📁 2. DATA DOWNLOAD + EXTRACTION
# ======================================================
def fetch_dataset():
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    archive = tf.keras.utils.get_file("facades.tar.gz", origin=url)
    parent = os.path.dirname(archive)
    target = os.path.join(parent, "facades")

    if not os.path.exists(target):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(parent)

    return target

dataset_dir = fetch_dataset()

print("📁 Train Samples:", len(os.listdir(dataset_dir + "/train")))
print("📁 Test Samples :", len(os.listdir(dataset_dir + "/test")))

# ======================================================
# 🖼️ 3. IMAGE LOADING + NORMALIZATION
# ======================================================
def read_and_preprocess(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw)

    width = tf.shape(img)[1] // 2
    tgt = img[:, :width, :]
    src = img[:, width:, :]

    tgt = tf.image.resize(tgt, [256, 256])
    src = tf.image.resize(src, [256, 256])

    src = (tf.cast(src, tf.float32) / 127.5) - 1
    tgt = (tf.cast(tgt, tf.float32) / 127.5) - 1

    return src, tgt

train_list = tf.data.Dataset.list_files(dataset_dir + "/train/*.jpg", shuffle=True)
test_list = tf.data.Dataset.list_files(dataset_dir + "/test/*.jpg", shuffle=True)

train_data = train_list.map(read_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(1)
test_data = test_list.map(read_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(1)

# ======================================================
# 🧱 4. BUILDING THE GENERATOR (U-NET)
# ======================================================
def encoder_block(filters, size, batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    seq = keras.Sequential()
    seq.add(layers.Conv2D(filters, size, strides=2, padding="same",
                          kernel_initializer=init, use_bias=False))
    if batchnorm:
        seq.add(layers.BatchNormalization())
    seq.add(layers.LeakyReLU())
    return seq

def decoder_block(filters, size, dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    seq = keras.Sequential()
    seq.add(layers.Conv2DTranspose(filters, size, strides=2, padding="same",
                                   kernel_initializer=init, use_bias=False))
    seq.add(layers.BatchNormalization())
    if dropout:
        seq.add(layers.Dropout(0.5))
    seq.add(layers.ReLU())
    return seq

def build_generator():
    inputs = layers.Input(shape=[256, 256, 3])

    downs = [
        encoder_block(64, 4, batchnorm=False),
        encoder_block(128, 4),
        encoder_block(256, 4),
        encoder_block(512, 4),
        encoder_block(512, 4),
        encoder_block(512, 4),
        encoder_block(512, 4),
        encoder_block(512, 4),
    ]

    ups = [
        decoder_block(512, 4, dropout=True),
        decoder_block(512, 4, dropout=True),
        decoder_block(512, 4, dropout=True),
        decoder_block(512, 4),
        decoder_block(256, 4),
        decoder_block(128, 4),
        decoder_block(64, 4),
    ]

    x = inputs
    skip_layers = []

    for d in downs:
        x = d(x)
        skip_layers.append(x)

    skip_layers = reversed(skip_layers[:-1])

    for u, s in zip(ups, skip_layers):
        x = u(x)
        x = layers.Concatenate()([x, s])

    final = layers.Conv2DTranspose(3, 4, strides=2, padding="same",
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                   activation="tanh")

    return keras.Model(inputs=inputs, outputs=final(x))

generator = build_generator()

# ======================================================
# 🧱 5. DISCRIMINATOR (PatchGAN)
# ======================================================
def build_discriminator():
    inp = layers.Input(shape=[256, 256, 3])
    tgt = layers.Input(shape=[256, 256, 3])
    combined = layers.concatenate([inp, tgt])

    init = tf.random_normal_initializer(0., 0.02)

    d1 = encoder_block(64, 4, batchnorm=False)(combined)
    d2 = encoder_block(128, 4)(d1)
    d3 = encoder_block(256, 4)(d2)

    pad1 = layers.ZeroPadding2D()(d3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False)(pad1)
    bn = layers.BatchNormalization()(conv)
    act = layers.LeakyReLU()(bn)

    pad2 = layers.ZeroPadding2D()(act)
    out = layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(pad2)

    return keras.Model(inputs=[inp, tgt], outputs=out)

discriminator = build_discriminator()

# ======================================================
# ⚙️ 6. LOSS FUNCTIONS + OPTIMIZERS
# ======================================================
bce = keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(disc_gen_out, gen_out, target):
    adv = bce(tf.ones_like(disc_gen_out), disc_gen_out)
    l1 = tf.reduce_mean(tf.abs(target - gen_out))
    return adv + (100 * l1)

def disc_loss(real_out, fake_out):
    real = bce(tf.ones_like(real_out), real_out)
    fake = bce(tf.zeros_like(fake_out), fake_out)
    return real + fake

gen_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)

# ======================================================
# 🏋️ 7. TRAINING LOOP
# ======================================================
@tf.function
def train_once(inp, tgt):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        g_pred = generator(inp, training=True)
        d_real = discriminator([inp, tgt], training=True)
        d_fake = discriminator([inp, g_pred], training=True)

        g_loss_value = gen_loss(d_fake, g_pred, tgt)
        d_loss_value = disc_loss(d_real, d_fake)

    g_grads = g_tape.gradient(g_loss_value, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss_value, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

# Train 1 Epoch (Demo)
for ep in range(1):
    print("🔄 Epoch 1 starting...")
    for img, lbl in train_data.take(50):
        train_once(img, lbl)
    print("✅ Epoch 1 finished")

# ======================================================
# 🎨 8. GENERATE OUTPUTS
# ======================================================
def visualize(model, inp, tgt):
    pred = model(inp, training=False)
    items = [inp[0], tgt[0], pred[0]]
    titles = ["Input", "Ground Truth", "Generated"]

    plt.figure(figsize=(14, 14))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(items[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()

for a, b in test_data.take(1):
    visualize(generator, a, b)

print("🎉 Pix2Pix Generation Completed Successfully!")
