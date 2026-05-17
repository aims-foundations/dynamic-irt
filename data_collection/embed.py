import argparse
import torch
import io
import pickle
import pandas as pd
from datasets import Dataset
from embed_text_package.embed_text_v2 import Embedder
from huggingface_hub import snapshot_download, HfApi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    
    upload_api = HfApi()
    data_folder = snapshot_download(
        "CodeInsightTeam/code_insights_csv", 
        repo_type="dataset",
    )
    main_data = pd.read_csv(f"{data_folder}/main_data.csv")
    
    embedder = Embedder()
    num_gpu = torch.cuda.device_count()
    embedder.load(
	    args.model,
        tensor_parallel_size=num_gpu,
    	enable_chunked_prefill=False,
    	enforce_eager=True,
        dtype=torch.float16
    )
    
    response = main_data["response"]
    response.fillna("", inplace=True)
    
    new_ds = {"text": response}
    text_dataset = Dataset.from_dict(new_ds)

    # Run the embeddings
    dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    embeddings = embedder.get_embeddings(dataloader, args.model, ["text"])
    output_dict = {
        "response_index": range(len(embeddings)),
        "embedding": embeddings
    }
    dataset = Dataset.from_dict(output_dict)
    dataset.push_to_hub(
        "CodeInsightTeam/code_insights_csv",
        "embeddings"
    )
    print("Embeddings uploaded")
    