from tkinter import *
import user_input as view
from image_file_manager import ImageFileControl
from training_dataset import TrainingDatasetImpl, TrainingDataset
from training_dataset_processor import ProcessingControl


def main():
    """Method initializes different objects involved in the execution of DatEx"""
    processing_control: ProcessingControl = ProcessingControl()
    training_dataset: TrainingDataset = TrainingDatasetImpl(processing_delegate=processing_control)
    dataset_file_control: ImageFileControl = ImageFileControl(training_dataset)

    root = Tk()
    # Set dimensions of the UI Frame.
    root.geometry("%dx%d+500+50" % (250, 690))
    view_model = view.MenuViewModel()

    view.MenuViewController(root, view_model, dataset_file_control, processing_control).mainloop()


if __name__ == "__main__":
    main()
