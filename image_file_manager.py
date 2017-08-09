"""
The image_file_manager module is responsible for managing the reading and saving of images
on the file system.
"""


from user_input import Preference, PreferenceDelegate, GeneralPreference, ProgressBarDelegate
import os
import image_manipulations as im
from training_dataset import TrainingDataset, TrainingDatasetVisitor, Example
from subprocess import call


class ImageFileControl(PreferenceDelegate, TrainingDatasetVisitor):
    """Controls the reading and saving of images and dispatches read images to the training_dataset module"""

    def __init__(self, training_dataset_delegate: TrainingDataset):
        self.training_dataset_delegate: TrainingDataset = training_dataset_delegate
        self.preference: GeneralPreference = None
        self.progress_bar_delegate: ProgressBarDelegate = None

    def receive_preference(self, preference: Preference):
        """Reads images and dispatches them to the training dataset and invokes visit of the training dataset"""
        self.preference = preference.general_preference

        images = ImageFileReadService.read_images(self.preference.source_path)

        # Initialize training dataset
        self.training_dataset_delegate.make_training_dataset(images)
        # Begin visit of the training dataset to save it (Processing is already completed at this point)
        self.training_dataset_delegate.accept(self)

        self.training_dataset_delegate.reset_training_dataset()
        self.progress_bar_delegate.done()

        call(['open', self.preference.target_path])

    def set_progress_bar_delegate(self, progress_bar_delegate: ProgressBarDelegate):
        self.progress_bar_delegate = progress_bar_delegate

    def visit(self, example: Example):
        label = example.get_label()
        image = example.get_image()

        ImageFileSaveService.save_image(label, image, self.preference.target_path)
        self.progress_bar_delegate.display_progress()


class ImageFileReadService:
    """Responsible for reading images of the file system"""

    @staticmethod
    def read_images(path: str):
        """Reads images from given path

        :param path: Folder location of the images
        :return: List of tuples, containing the image and the image name
        """
        images = []
        files = os.listdir(path)

        for file in files:
            elements_file_name = file.split(".")

            i = 0
            if len(elements_file_name) == 2:
                i = 1
            elif len(elements_file_name) == 3:
                i = 2

            if elements_file_name[i] != "jpg" and elements_file_name[i] != "png" and elements_file_name[i] != "JPG":
                print("%s is not a valid image file." % file)
                continue

            image_path = "%s/%s" % (path, file)
            piece_name = elements_file_name[0]
            image = im.read_image(image_path)

            images.append((piece_name, image))

        return images


class ImageFileSaveService:

    @staticmethod
    def save_image(label: str, image, target_path: str):
        """Saves image to given path"""
        number = 0
        while os.path.isfile("%s/%s.png" % (target_path, "%s%s" % (label, '{0:05}'.format(number)))):
            number += 1

        im.save_image(path=("%s/%s.png" % (target_path, "%s%s" % (label, '{0:05}'.format(number)))), image=image)
