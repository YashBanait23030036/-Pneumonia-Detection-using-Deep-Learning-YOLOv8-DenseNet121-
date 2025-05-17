from pipeline_code.config import Config
from pipeline_code.dataset import get_dataloaders
from pipeline_code.model import PneumoniaModel
from pipeline_code.train import train_model
from pipeline_code.evaluate import evaluate_model

def run_pipeline():
    config = Config()

    train_loader, val_loader = get_dataloaders(config)

    model = PneumoniaModel(config)

    trained_model = train_model(model, train_loader, val_loader, config)

    evaluate_model(trained_model, val_loader, config)
    print("✅ Pipeline completed successfully!")

run_pipeline()
