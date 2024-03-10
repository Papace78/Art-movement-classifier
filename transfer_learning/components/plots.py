import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf

def learning_curves(history, title = 'myman.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(acc,label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epcoh')

    plt.savefig(title)
    plt.close()

def confusion_matrix(model, val_ds):

    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in val_ds:
        # append true labels
        y_true.append(np.argmax(label_batch, axis=1))
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)

    conf_mat = tf.math.confusion_matrix(correct_labels, predicted_labels)
    conf_mat = conf_mat.numpy() / conf_mat.numpy().sum(axis=1)[:, np.newaxis]

    class_names = sorted(os.listdir(os.path.join("raw_data","wikiart","trainval_directory")))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    heatmap = sns.heatmap(conf_mat, annot=True, cmap="Blues", ax=ax, cbar=False,
                xticklabels=class_names, yticklabels=class_names, fmt='.2%')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_ylabel("Labels")
    ax.set_xlabel("Prediction")
    heatmap.savefig('test')
