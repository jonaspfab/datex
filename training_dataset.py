from abc import ABC, abstractmethod


class Example(ABC):

    @abstractmethod
    def get_image(self):
        pass

    @abstractmethod
    def set_image(self, image):
        pass

    @abstractmethod
    def get_label(self) -> str:
        pass


class TrainingDatasetVisitor(ABC):

    @abstractmethod
    def visit(self, example: Example):
        pass
    

class TrainingDataset(ABC):

    @abstractmethod
    def make_dataset(self, images):
        pass

    def accept(self, visitor: TrainingDatasetVisitor):
        pass

    def add_example(self, label: str, image):
        pass

    def count(self):
        pass


class TrainingSetDelegate(ABC):

    @abstractmethod
    def receive_dataset(self, dataset: TrainingDataset):
        pass


class ExampleImpl(Example):

    def __init__(self, label: str, image):
        self.label = label
        self.image = image

    def accept(self, visitor: TrainingDatasetVisitor):
        visitor.visit(self)

    def get_image(self):
        return self.image

    def get_label(self) -> str:
        return self.label

    def set_image(self, image):
        self.image = image


class TrainingDatasetImpl(TrainingDataset):

    def __init__(self, processing_delegate: TrainingSetDelegate):
        self.examples: [Example] = []
        self.dataset_file_delegate: TrainingSetDelegate = None
        self.processing_delegate: TrainingSetDelegate = processing_delegate

    def make_dataset(self, images):
        for image in images:
            label = image[0]
            image = image[1]

            new_example = ExampleImpl(label, image)
            self.examples.append(new_example)

        self.processing_delegate.receive_dataset(self)

    def accept(self, visitor: TrainingDatasetVisitor):
        examples = list(self.examples)

        for example in examples:
            example.accept(visitor)

    def add_example(self, label: str, image):
        new_example = ExampleImpl(label, image)

        self.examples.append(new_example)

    def count(self):
        return len(self.examples)
