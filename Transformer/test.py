from Transformer.model import transformer_model
from Transformer.utils import*
from Transformer.Config import*

import os


import random
import matplotlib.pyplot as plt
import zipfile

epochs = Parameters.get("epochs")
batch_size = Parameters.get("batch_size")
val_size = Parameters.get("val_size")

head_size = Parameters.get("head_size")
num_heads = Parameters.get("num_heads")
ff_dim = Parameters.get("ff_dim")
num_encoder_block = Parameters.get("num_encoder_block")
mlp_units = Parameters.get("mlp_units")
dropout = Parameters.get("dropout")
n_seq_in = Parameters.get("n_seq_in")
n_features = Parameters.get("n_features")
n_seq_out = Parameters.get("n_seq_out")
features = Parameters.get("features")


load_model(transformer_model, "Transformer/trained_model/transformer.pth")



def evalTransformer():
    data_zip = zipfile.ZipFile("data/test_data/company_f.zip")
    random_csv_file = random.choice(data_zip.namelist())
    
    try:
        with data_zip.open(random_csv_file) as f:
            test_data_csv = pd.read_csv(f)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            transformer_model.to(device)
            transformer_model.eval()  # Set model to evaluation mode
            if test_data_csv.shape[0] > 15:
                data = data_extractor(test_data_csv, val_size, n_seq_in, n_features, n_seq_out, "DlyPrc")
                evalY_tensor = torch.tensor(data.y_eval, dtype=torch.float32).to(device)
                data.normalzie_data()
                evalX_tensor = torch.tensor(data.X_eval, dtype=torch.float32).to(device)
                print(transformer_model.parameters)
                with torch.no_grad():  # Temporarily disable gradient calculation
                    predictions = transformer_model(evalX_tensor)
                
                predictions = data.denormalzie_evaldata(predictions.cpu())
                predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
                l = torch.mean((torch.abs(predictions - evalY_tensor) / evalY_tensor))
                print("Finished evaluation on file:", random_csv_file)
                print("Loss:", l.item())
    except Exception as e:
        print(f"Error processing {random_csv_file}: {e}")
    finally:
        data_zip.close()

    return
