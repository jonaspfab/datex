"""
The training_dataset_processor module is responsible for expanding a training dataset,
by generating new examples which are modifications of existing examples of the set.
"""

from training_dataset import TrainingDataset, TrainingSetDelegate, TrainingDatasetVisitor, Example
from user_input import Preference, GeneralPreference, ModificationPreference, PreferenceDelegate, ProgressBarDelegate
import random
import image_manipulations as im


class ProcessingControl(TrainingSetDelegate, PreferenceDelegate):

    def __init__(self):
        self.preference: Preference = None
        self.progress_bar_delegate: ProgressBarDelegate = None

    def receive_dataset(self, training_dataset: TrainingDataset):
        expansion_service = DuplicationService(training_dataset, self.preference.general_preference)
        modification_service = ModificationService(self.preference.modification_preference, self.progress_bar_delegate)

        training_dataset.accept(expansion_service)

        self.progress_bar_delegate.set_progress_step(training_dataset.count())

        training_dataset.accept(modification_service)

    def receive_preference(self, preference: Preference):
        self.preference = preference

    def set_progress_bar_delegate(self, progress_bar_delegate: ProgressBarDelegate):
        self.progress_bar_delegate = progress_bar_delegate


class DuplicationService(TrainingDatasetVisitor):
    def __init__(self, dataset: TrainingDataset, preferences: GeneralPreference):
        self.dataset = dataset
        self.preferences = preferences

    def visit(self, example: Example):
        number_of_images = self.preferences.number_of_images
        examples = self.duplicate_example(number_of_images - 1, example)

        for ex in examples:
            self.dataset.add_example(ex[0], ex[1])

    @staticmethod
    def duplicate_example(number_of_images: int, example: Example):
        examples = []

        label: str = example.get_label()
        image = example.get_image()

        for i in range(0, number_of_images):
            examples.append((label, image))

        return examples


class ModificationInformation:

    def __init__(self, image, border_proportion: int):
        self.image = image
        self.border_proportion: int = border_proportion
        self.original_dimensions: int = image.shape[0]
        self.border_size = 2 * int(self.original_dimensions / border_proportion)
        self.bounding_box_dimensions: int = self.original_dimensions - self.border_size
        self.proportion_enlarged_original: float = 0

    def update(self, image):
        self.original_dimensions = image.shape[0]
        self.border_size = 2 * int(self.original_dimensions / self.border_proportion)
        self.bounding_box_dimensions = self.original_dimensions - self.border_size
        self.proportion_enlarged_original: float = 0

    def set_proportion_enlarged_original(self, image):
        new_dimensions = image.shape[0]
        self.proportion_enlarged_original = new_dimensions / self.original_dimensions

    def update_after_zoom_transformation(self, new_dimensions: int, original_dimensions: int):
        proportion = new_dimensions / original_dimensions

        self.bounding_box_dimensions = int(self.bounding_box_dimensions * proportion)

    def get_enlarged_bounding_box_dimensions(self):
        return int(self.bounding_box_dimensions * self.proportion_enlarged_original)


class ModificationService(TrainingDatasetVisitor):
    def __init__(self, preference: ModificationPreference, progress_bar_delegate: ProgressBarDelegate):
        self.preference: ModificationPreference = preference
        self.progress_bar_delegate: ProgressBarDelegate = progress_bar_delegate
        self.modification_information: ModificationInformation = None
        self.image_counter = 0

    def visit(self, example: Example):
        image = example.get_image()
        self.modification_information = ModificationInformation(image, self.preference.border_proportion)
        self.pre_process_image()

        if self.preference.should_rotate:
            self.apply_rotation_transformation()
        if self.preference.should_rotate_gradually:
            self.apply_gradual_rotation_transformation()
        if self.preference.should_zoom_out and self.preference.should_zoom_in:
            random_number = random.random()
            self.apply_zoom_out_transformation() if random_number < 0.5 else self.apply_zoom_in_transformation()
        elif self.preference.should_zoom_out:
            self.apply_zoom_out_transformation()
        elif self.preference.should_zoom_in:
            self.apply_zoom_in_transformation()
        if self.preference.should_off_center:
            self.apply_off_center_transformation()

        self.post_process_image()

        if self.preference.should_adjust_lighting:
            self.apply_lighting_adjustment()
        if self.preference.should_blur:
            self.apply_blur_transformation()

        image = self.modification_information.image
        example.set_image(image)
        self.image_counter += 1
        self.progress_bar_delegate.display_progress()

    def pre_process_image(self):
        image = self.modification_information.image
        original_dimensions = image.shape[0]

        new_dimensions = self.preference.dimensions * 2
        if new_dimensions < original_dimensions:
            image = im.set_dimensions(image, new_dimensions)
            original_dimensions = image.shape[0]
            self.modification_information.update(image)

        border_size = self.modification_information.border_size

        x_0, y_0 = border_size, border_size
        x_1, y_1 = original_dimensions - border_size, original_dimensions - border_size

        image = im.enlarge_background(image, (original_dimensions * 3), x_0, y_0, x_1, y_1)
        self.modification_information.set_proportion_enlarged_original(image)

        self.modification_information.image = image

    def apply_rotation_transformation(self):
        image = self.modification_information.image

        min = 0
        max = self.preference.rotation_angle

        angle = int(min + random.random() * (max - min) + 1)
        transformed_image = im.rotate(image, angle)

        self.modification_information.image = transformed_image

    def apply_gradual_rotation_transformation(self):
        image = self.modification_information.image

        angle = self.preference.gradual_rotation_angle * self.image_counter
        transformed_image = im.rotate(image, angle)

        self.modification_information.image = transformed_image

    def apply_zoom_out_transformation(self):
        image = self.modification_information.image

        min = image.shape[0]
        max = int(min / 1.5)

        original_dimensions = image.shape[0]
        new_dimensions = int(min + random.random() * (max - min) + 1)
        transformed_image = im.zoom_out(image, new_dimensions)
        self.modification_information.update_after_zoom_transformation(new_dimensions, original_dimensions)

        self.modification_information.image = transformed_image

    def apply_zoom_in_transformation(self):
        image = self.modification_information.image

        min = image.shape[0]
        max = self.modification_information.get_enlarged_bounding_box_dimensions()

        original_dimensions = image.shape[0]
        new_dimensions = int(min + random.random() * (max - min) + 1)
        transformed_image = im.zoom_in(image, new_dimensions)
        self.modification_information.update_after_zoom_transformation(original_dimensions, new_dimensions)

        self.modification_information.image = transformed_image

    def apply_off_center_transformation(self):
        image = self.modification_information.image

        information = self.modification_information
        # Determines the limit of how much the piece can be shifted left/up
        min = int((information.original_dimensions - information.bounding_box_dimensions) / 2)
        # Determines the limit of how much the piece can be shifted right/down
        max = -min

        # Calculate random values in between min and max
        horizontal_shift = int(min + random.random() * (max - min) + 1)
        vertical_shift = int(min + random.random() * (max - min) + 1)

        transformed_image = im.off_center(image, horizontal_shift, vertical_shift)

        self.modification_information.image = transformed_image

    def post_process_image(self):
        image = self.modification_information.image

        original_dimensions = self.modification_information.original_dimensions
        post_processed_image = im.crop_image(image, original_dimensions)

        new_dimensions = self.preference.dimensions
        post_processed_image = im.set_dimensions(post_processed_image, new_dimensions)

        if self.preference.should_grayscale:
            post_processed_image = im.grayscale(post_processed_image)

        self.modification_information.image = post_processed_image

    def apply_lighting_adjustment(self):
        image = self.modification_information.image

        min = -self.preference.lighting_factor
        max = self.preference.lighting_factor

        contrast_factor = int(min + random.random() * (max - min) + 1)
        exposure_factor = int(min + random.random() * (max - min) + 1)
        brightness_factor = int(min + random.random() * (max - min) + 1)
        transformed_image = im.adjust_contrast_exposure(image, contrast_factor, exposure_factor)
        transformed_image = im.adjust_brightness(transformed_image, brightness_factor)

        self.modification_information.image = transformed_image

    def apply_blur_transformation(self):
        image = self.modification_information.image

        min = 0
        max = self.preference.blur_factor

        blur_factor = int(min + random.random() * (max - min) + 1)
        transformed_image = im.blur(image, blur_factor)

        self.modification_information.image = transformed_image
