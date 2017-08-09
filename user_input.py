"""
The user_input module is responsible for handling and tracking all user inputs.
In particular, it realizes DatEx's user interface, tracks all, by the user chosen,
preference and provides them to other modules.
"""

from tkinter import *
from tkinter import filedialog
from abc import ABC, abstractmethod


class GeneralPreference:
    """Contains general information relevant to DatEx"""

    def __init__(self, source_path: str, target_path: str, number_of_images: int):
        # Defines the path to the images used to setup the base training dataset
        self.source_path: str = source_path
        # Defines the path where the expanded training dataset will be saved to
        self.target_path: str = target_path
        # Defines how many images are generated of each image from the base training dataset
        self.number_of_images: int = number_of_images


class ModificationPreference:
    """Contains information regarding the modifications that are applied to images of training dataset"""

    def __init__(self, dimensions: int, should_rotate: bool, rotation_angle: int, should_rotate_gradually: bool,
                 gradual_rotation_angle: int, should_off_center: bool, should_zoom_out: bool, should_zoom_in: bool,
                 should_blur: bool, blur_factor: int, should_adjust_lighting: bool, lighting_factor: int,
                 should_grayscale: bool, border_proportion: bool):

        # Dimensions of the generated images
        self.dimensions: int = dimensions
        self.should_rotate: bool = should_rotate
        # Restricts the range of possible rotation angles with an upper boundary
        self.rotation_angle: int = rotation_angle
        self.should_rotate_gradually: bool = should_rotate_gradually
        self.gradual_rotation_angle: bool = gradual_rotation_angle
        self.should_off_center: bool = should_off_center
        self.should_zoom_out: bool = should_zoom_out
        self.should_zoom_in: bool = should_zoom_in
        self.should_blur: bool = should_blur
        self.blur_factor: int = blur_factor
        self.should_adjust_lighting: bool = should_adjust_lighting
        # Value determining how strongly the lighting adjustments will be
        self.lighting_factor: int = lighting_factor
        self.should_grayscale: bool = should_grayscale
        # Defines the proportion of the image which is border (not containing the object)
        self.border_proportion: int = border_proportion


class Preference:
    """Contains all information necessary for the execution of DatEx.

    Consists of the ModificationPreference and the GeneralPreference.
    """

    def __init__(self, general_preference: GeneralPreference, modification_preference: ModificationPreference):
        self.general_preference: GeneralPreference = general_preference
        self.modification_preference: ModificationPreference = modification_preference


class ProgressBarDelegate(ABC):
    """Provides methods to present the progress of the application on the UI.

    (Not relevant for DatEx's functionality)
    """

    @abstractmethod
    def set_progress_step(self, progress_step: float):
        """Set the size of each progress step.

        The progress step determines the amount by which the progress is increased
        for each display_progress() method call.
        """
        pass

    @abstractmethod
    def display_progress(self):
        """Add a progress step to the progress and display it on the UI"""
        pass


class PreferenceDelegate(ABC):
    """Provides methods to realize the exchange of preference to other modules"""

    @abstractmethod
    def receive_preference(self, preference: Preference):
        pass

    @abstractmethod
    def set_progress_bar_delegate(self, progress_bar_delegate: ProgressBarDelegate):
        """Realizes the injection of a dependency to an object displaying the application's progress.

        (Not relevant for DatEx's functionality)
        """
        pass


class MenuViewModel:
    """Tracks all information displayed in the menu view"""

    def __init__(self):
        # General preference
        self.source_path: StringVar = StringVar()
        self.target_path: StringVar = StringVar()
        self.number_of_images: IntVar = IntVar()

        # Modification preference
        self.dimensions: IntVar = IntVar()
        self.should_rotate: IntVar = IntVar()
        self.rotation_angle: IntVar = IntVar()
        self.rotation_angle.set(360)
        self.should_rotate_gradually: IntVar = IntVar()
        self.should_off_center: IntVar = IntVar()
        self.should_zoom_out: IntVar = IntVar()
        self.should_zoom_in: IntVar = IntVar()
        self.blur_factor: IntVar = IntVar()
        self.lighting_factor: IntVar = IntVar()
        self.should_grayscale: IntVar = IntVar()
        self.border_proportion: IntVar = IntVar()
        self.border_proportion.set(10)


class MenuViewController(Frame, ProgressBarDelegate):
    """Sets up view elements and bindings between the menu view and the MenuViewModel"""

    def __init__(self, parent, view_model: MenuViewModel, image_file_delegate: PreferenceDelegate,
                 processing_delegate: PreferenceDelegate):
        Frame.__init__(self, parent)

        parent.title("DatEx")
        self.pack(fill=BOTH, expand=1)

        self.view_model = view_model
        self.image_file_delegate: PreferenceDelegate = image_file_delegate
        self.processing_delegate: PreferenceDelegate = processing_delegate

        y_base = 90
        x_labels = 10
        x_checkboxes = 150

        title_bg = Canvas(self, bg="#596A7B", height=70, width=250, borderwidth=0, highlightthickness=0)
        title_bg.place(x=0, y=0)
        title_label = Label(self, text="DatEx", bg="#596A7B", fg="#fff", font=("Helvetica", 20))
        title_label.place(x=125, y=23)

        logo_image = PhotoImage(file="./assets/logo.gif")
        logo_image = logo_image.subsample(6, 6)
        logo_label = Label(self, image=logo_image, borderwidth=0, highlightthickness=0)
        logo_label.image = logo_image
        logo_label.place(x=50, y=9)

        self.progress_bar = Canvas(self, bg="#EC8B1A", height=10, width=270)
        self.progress_bar.place(x=-275, y=640)
        self.progress_step: float = 0
        self.progress: float = 0

        self.number_of_images_label = Label(self, text="Number of Images")
        self.number_of_images_label.place(x=x_labels, y=y_base)
        self.number_of_images_entry = Entry(self, width=7, textvariable=view_model.number_of_images)
        self.number_of_images_entry.place(x=x_checkboxes, y=y_base)

        self.border_proportion_label = Label(self, text="Border\nProportion", justify=LEFT)
        self.border_proportion_label.place(x=x_labels, y=y_base + 33)
        self.border_proportion_scale = Scale(self, from_=4, to_=99, orient=HORIZONTAL, showvalue=0, length=80,
                                             variable=view_model.border_proportion)
        self.border_proportion_scale.place(x=x_checkboxes + 4, y=y_base + 42)
        self.border_proportion_amount_label = Label(self, textvariable=view_model.border_proportion, anchor=W, width=2)
        self.border_proportion_amount_label.place(x=x_checkboxes - 20, y=y_base + 40)
        self.border_proportion_fraction_label = Label(self, text="1/")
        self.border_proportion_fraction_label.place(x=x_checkboxes - 35, y=y_base + 40)

        self.grayscale_label = Label(self, text="Grayscale")
        self.grayscale_label.place(x=x_labels, y=y_base + 80)
        self.grayscale_checkbutton = Checkbutton(self, variable=view_model.should_grayscale)
        self.grayscale_checkbutton.place(x=x_checkboxes, y=y_base + 80)

        self.blurring_label = Label(self, text="Blur Images")
        self.blurring_label.place(x=x_labels, y=y_base + 120)
        self.blurring_scale = Scale(self, from_=0, to_=30, orient=HORIZONTAL, showvalue=0, length=80,
                                    variable=view_model.blur_factor)
        self.blurring_scale.place(x=x_checkboxes+4, y=y_base + 122)
        self.blurring_amount_label = Label(self, textvariable=view_model.blur_factor, anchor=E, width=3)
        self.blurring_amount_label.place(x=x_checkboxes-30, y=y_base+120)

        self.lighting_label = Label(self, text="Adjust Lighting")
        self.lighting_label.place(x=x_labels, y=y_base + 160)
        self.lighting_scale = Scale(self, from_=0, to_=30, orient=HORIZONTAL, showvalue=0, length=80,
                                    variable=view_model.lighting_factor)
        self.lighting_scale.place(x=x_checkboxes + 4, y=y_base + 162)
        self.lighting_amount_label = Label(self, textvariable=view_model.lighting_factor, anchor=E, width=3)
        self.lighting_amount_label.place(x=x_checkboxes - 30, y=y_base + 160)

        self.rotation_label = Label(self, text="Rotate Images")
        self.rotation_label.place(x=x_labels, y=y_base + 200)
        self.rotation_checkbutton = Checkbutton(self, variable=view_model.should_rotate)
        self.rotation_checkbutton.place(x=x_checkboxes, y=y_base + 200)
        self.rotation_angle_entry = Entry(self, width=3, textvariable=view_model.rotation_angle, justify=RIGHT)
        self.rotation_angle_entry.place(x=x_checkboxes + 30, y=y_base + 200)
        self.rotation_angle_symbol_label = Label(self, text="°")
        self.rotation_angle_symbol_label.place(x=x_checkboxes + 67, y=y_base + 200)

        self.rotation_gradually_label = Label(self, text="Rotate Gradually")
        self.rotation_gradually_label.place(x=x_labels, y=y_base + 240)
        self.rotation_gradually_checkbutton = Checkbutton(self, variable=view_model.should_rotate_gradually)
        self.rotation_gradually_checkbutton.place(x=x_checkboxes, y=y_base + 240)

        self.zoom_out_label = Label(self, text="Zoom Out Images")
        self.zoom_out_label.place(x=x_labels, y=y_base + 280)
        self.zoom_out_checkbutton = Checkbutton(self, variable=view_model.should_zoom_out)
        self.zoom_out_checkbutton.place(x=x_checkboxes, y=y_base + 280)

        self.zoom_out_label = Label(self, text="Zoom In Images")
        self.zoom_out_label.place(x=x_labels, y=y_base + 320)
        self.zoom_out_checkbutton = Checkbutton(self, variable=view_model.should_zoom_in)
        self.zoom_out_checkbutton.place(x=x_checkboxes, y=y_base + 320)

        self.off_centering_label = Label(self, text="Off Center Images")
        self.off_centering_label.place(x=x_labels, y=y_base + 360)
        self.off_centering_checkbutton = Checkbutton(self, variable=view_model.should_off_center)
        self.off_centering_checkbutton.place(x=x_checkboxes, y=y_base + 360)

        self.dimensions_label = Label(self, text="Dimensions")
        self.dimensions_label.place(x=x_labels, y=y_base + 400)
        self.dimensions_entry = Entry(self, width=4, textvariable=view_model.dimensions)
        self.dimensions_entry.place(x=x_checkboxes, y=y_base + 397)
        self.dimensions_y_label = Label(self, textvariable=view_model.dimensions, width=4, anchor=E)
        self.dimensions_y_label.place(x=x_checkboxes - 50, y=y_base + 400)
        self.dimensions_x_label = Label(self, text="x")
        self.dimensions_x_label.place(x=x_checkboxes - 11, y=y_base + 400)

        self.source_path_button = Button(self, text="Source", command=self.choose_source_path)
        self.source_path_button.place(x=x_labels, y=y_base + 440)
        self.source_path_label = Label(self, textvariable=view_model.source_path, width=18, anchor=E, justify=RIGHT)
        self.source_path_label.place(x=x_checkboxes-68, y=y_base+443)

        self.target_path_button = Button(self, text="Target", command=self.choose_target_path)
        self.target_path_button.place(x=x_labels, y=y_base + 480)
        self.target_path_label = Label(self, textvariable=view_model.target_path, width=18, anchor=E)
        self.target_path_label.place(x=x_checkboxes - 68, y=y_base + 483)

        self.run_button = Button(self, text="Run", command=self.executeGArD)
        self.run_button.place(x=100, y=y_base + 510)

    def choose_source_path(self):
        """Gets invoked by the pressing the 'Source' button and opens a file dialog"""
        path = filedialog.askdirectory()
        self.view_model.source_path.set(path)

    def choose_target_path(self):
        """Gets invoked by the pressing the 'Target' button and opens a file dialog"""
        path = filedialog.askdirectory()
        self.view_model.target_path.set(path)

    def set_progress_step(self, number_of_examples: float):
        self.progress_step = 50 / number_of_examples

    def display_progress(self):
        """Moves progress bar to the right"""
        self.progress += self.progress_step
        if self.progress == 100:
            self.progress_bar.place(x=-275, y=640)
        else:
            self.progress_bar.place(x=self.progress*2.55-275, y=640)
        self.progress_bar.update()

    def executeGArD(self):
        """Initialize Preference objects and send them to the training_dataset and image_file_manager modules"""

        # General Preference
        source_path = self.view_model.source_path.get()
        target_path = self.view_model.target_path.get()
        number_of_images = self.view_model.number_of_images.get()

        general_preference = GeneralPreference(source_path, target_path, number_of_images)

        # Modification Preference
        dimensions = self.view_model.dimensions.get()
        should_rotate = self.view_model.should_rotate.get() == 1
        rotation_angle = self.view_model.rotation_angle.get()
        should_rotate_gradually = self.view_model.should_rotate_gradually.get() == 1
        gradual_rotation_angle = int(360 / self.view_model.number_of_images.get())
        should_off_center = self.view_model.should_off_center.get() == 1
        should_zoom_out = self.view_model.should_zoom_out.get() == 1
        should_zoom_in = self.view_model.should_zoom_in.get() == 1
        should_blur = self.view_model.blur_factor.get() > 0
        blur_factor = self.view_model.blur_factor.get()
        should_adjust_lighting = self.view_model.lighting_factor.get() > 0
        lighting_factor = self.view_model.lighting_factor.get()
        should_grayscale = self.view_model.should_grayscale.get() == 1
        border_proportion = self.view_model.border_proportion.get()

        modification_preference = ModificationPreference(dimensions, should_rotate, rotation_angle,
                                                         should_rotate_gradually, gradual_rotation_angle,
                                                         should_off_center, should_zoom_out, should_zoom_in,
                                                         should_blur, blur_factor, should_adjust_lighting,
                                                         lighting_factor, should_grayscale, border_proportion)

        preference: Preference = Preference(general_preference, modification_preference)

        self.processing_delegate.set_progress_bar_delegate(self)
        self.image_file_delegate.set_progress_bar_delegate(self)
        self.processing_delegate.receive_preference(preference)
        self.image_file_delegate.receive_preference(preference)
