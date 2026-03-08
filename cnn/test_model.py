import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from cnn.cnn import get_data, get_ds

def main():
    CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    (train_images, train_labels), (test_images, test_labels), (validation_images, validation_labels) = get_data()
    train_ds, validation_ds, test_ds = get_ds(train_images, train_labels, validation_images, validation_labels, test_images, test_labels)

    # model_path = "models/alexnet_best.keras"
    model_path = "models/alexnet_final.keras"
    model = keras.models.load_model(model_path)

    model.evaluate(test_ds)

    predictions = model.predict(test_ds)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_labels.flatten()

    cm = confusion_matrix(true_labels, predicted_labels)

    print(cm)
    plot_confusion_matrix(cm, CLASS_NAMES, model_path)


def plot_confusion_matrix(confusion_matrix, class_names, title):
    title_match = re.findall(r"_([^.]+)\.", title)
    title = title_match[0]

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({title} - 30 epochs)")
    plt.tight_layout()

    plt.savefig(f"alexnet/confusion_matrix_{title}_30.png")
    plt.show()


if __name__=="__main__":
    main()