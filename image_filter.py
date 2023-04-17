import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image


def filter_human_images(annotations_path, output_path):
    human_images = []
    non_human_images = []

    for annotation_file in os.listdir(annotations_path):
        tree = ET.parse(os.path.join(annotations_path, annotation_file))
        root = tree.getroot()
        image_filename = root.find('filename').text

        objects = root.findall('object')
        has_human = False
        for obj in objects:
            if obj.find('name').text == 'person':
                has_human = True
                break

        if has_human:
            human_images.append(image_filename)
        else:
            non_human_images.append(image_filename)

    return human_images, non_human_images


def prepare_dataset(src_path, dst_path, human_images, non_human_images, target_size=(200, 200)):
    os.makedirs(os.path.join(dst_path, 'human'))
    os.makedirs(os.path.join(dst_path, 'non_human'))

    for img in human_images:
        im = Image.open(os.path.join(src_path, img))
        im = im.resize(target_size)
        im.save(os.path.join(dst_path, 'human', img))

    for img in non_human_images:
        im = Image.open(os.path.join(src_path, img))
        im = im.resize(target_size)
        im.save(os.path.join(dst_path, 'non_human', img))


def main():
    annotations_path = 'VOC2012_train_val/Annotations'
    src_path = 'VOC2012_train_val/JPEGImages'
    dst_path = 'human_dataset'

    human_images, non_human_images = filter_human_images(annotations_path, dst_path)
    prepare_dataset(src_path, dst_path, human_images, non_human_images)


if __name__ == '__main__':
    main()
