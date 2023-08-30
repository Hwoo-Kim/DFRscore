import argparse

from scripts.modelScripts.model import DFRscore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--smiles", type=str)
    args = parser.parse_args()

    dfr_model = DFRscore.from_trained_model(args.model_path)
    dfrscore = dfr_model.smiToScore(args.smiles)

    print(dfrscore)
