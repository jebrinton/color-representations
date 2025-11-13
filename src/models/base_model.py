from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def load(self): pass

    @abstractmethod
    def get_hidden_activations(self, text: str): pass
