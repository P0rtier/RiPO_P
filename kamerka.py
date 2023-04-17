import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pandas as pd

def load_models_and_labels():
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']
    return detector, labels

def capture_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def process_video(detector, labels, cap):
    while True:
        ret, rgb = cap.read()
        rgb_tensor = tf.expand_dims(tf.convert_to_tensor(rgb, dtype=tf.uint8), 0)
        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_labels = [labels[i] for i in classes.numpy().astype('int')[0]]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue

            img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_boxes, f'{100 * round(score, 0)}', (xmax, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('black and white', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    detector, labels = load_models_and_labels()
    cap = capture_video()
    process_video(detector, labels, cap)


if __name__ == '__main__':
    main()