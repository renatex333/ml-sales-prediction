import sys
from src.trainer import train

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: python train.py <path to sql script>")
        sys.exit(1)
    else:
        query_path = sys.argv[-1]
        train(query_path)
        sys.exit(0)