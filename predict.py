import sys
from src.predictor import predict

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python predict.py <path to model> <path to sql script>")
        sys.exit(1)
    else:
        data_path = sys.argv[-1]
        model_path = sys.argv[-2]
        predict(data_path, model_path)
        sys.exit(0)