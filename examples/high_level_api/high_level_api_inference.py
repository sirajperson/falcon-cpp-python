import json
import argparse

from falcon_cpp import Falcon

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../../models/tiiuae_falcon-7b/ggml-model-tiiuae_falcon-7b-f16.bin")
args = parser.parse_args()

llm = Falcon(model_path=args.model)

output = llm(
    "Question: What are the names of the planets in the solar system? Answer: ",
    max_tokens=48,
    stop=["Q:", "\n"],
    echo=True,
)

print(json.dumps(output, indent=2))
