import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from model import Retinexformer
from dataset import get_dataloaders


def load_model(model_path, device):
    model = Retinexformer().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path,
                                         map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"model not found at {model_path}")
    model.eval()
    return model


def test(model, test_loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, names) in enumerate(tqdm(test_loader,
                                                   desc="Testing")):
            inputs = inputs.to(device)
            outputs, illum = model(inputs)

            output_path = os.path.join(save_dir, names[0])
            save_image(outputs, output_path)



def main(root_dir='dataset',
         model_path='Model/Model.yml',
         save_dir='Output',
         num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    _, test_loader = get_dataloaders(root_dir=root_dir,
                                     batch_size=1,
                                     num_workers=num_workers)

    # Load model
    model = load_model(model_path, device)

    # Run testing
    test(model, test_loader, save_dir, device)
    print(f"Testing done. Results saved in {save_dir}")


if __name__ == '__main__':
    main(root_dir='dataset',
         model_path='Model/Model.yml',
         save_dir='Output',
         num_workers=2)
