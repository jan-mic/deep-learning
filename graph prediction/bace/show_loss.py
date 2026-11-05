from train import generate_charts

CSV_PATH = "/home/jd/PycharmProjects/deep-learning/task3/bace/logs/bace_Transformer_L3_mlp_dim1/version_2/metrics.csv"
EXP_NAME = "wykres"
Y_MIN = 0.4
Y_MAX = 0.8

if __name__ == "__main__":
    generate_charts(CSV_PATH, EXP_NAME, y_min=Y_MIN, y_max=Y_MAX)
