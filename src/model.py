import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.3): #-> None:

        super().__init__()
        # -> 3x224x224
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # -> 16x224x224
            #nn.BatchNorm2d(16),            
            nn.ReLU(),  
            nn.MaxPool2d(2, 2), # -> 16x112x112
            #nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, 3, padding=1), # -> 32x112x112 
            nn.BatchNorm2d(32),       
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x56x56
            #nn.Dropout2d(dropout),
            
            nn.Conv2d(32, 64, 3, padding=1), # -> 64x56x56
            nn.BatchNorm2d(64),        
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            #nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, 3, padding=1), # -> 128x28x28
            nn.BatchNorm2d(128),           
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
            #nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, 3, padding=1), # -> 256x14x14
            nn.BatchNorm2d(256),           
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256x7x7
            #nn.Dropout2d(dropout),
            
            nn.Flatten(),  # -> 1x256x7x7=12544
            
            nn.Linear(256 * 7 * 7, 6000),            
            #nn.BatchNorm1d(6000),
            nn.Dropout(dropout),
            nn.ReLU(),   
            
            
            nn.Linear(6000, 3000),
            #nn.BatchNorm1d(3000),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            
            nn.Linear(3000, num_classes)        
      
        )

    def forward(self, x: torch.Tensor): #-> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
