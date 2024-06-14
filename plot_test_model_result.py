import pandas as pd
import matplotlib.pyplot as plt

# run test_qa.py first, then this model will plot the result of test model

def plot_results(csv_file: str, output_file: str) -> None:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_file)

    # Tạo figure và axis
    fig, ax1 = plt.subplots()

    # Thiết lập vị trí các cột và độ rộng của các cột
    x = range(len(df))
    width = 0.4

    # Tạo biểu đồ cột cho loss
    ax1.bar([i - width/2 for i in x], df['loss'], width=width, label='Loss', color='blue', alpha=0.6)
    ax1.set_ylabel('Values')
    ax1.tick_params(axis='y')

    # Tạo biểu đồ cột cho accuracy trên cùng trục Y
    ax1.bar([i + width/2 for i in x], df['accuracy'], width=width, label='Accuracy', color='orange', alpha=1)

    # Thiết lập tên trục x
    plt.xticks(x, df['model_type'], ha='center')

    # Thiết lập tiêu đề
    plt.title('Loss and Accuracy of Different Models')

    # Thêm chú thích (legend)
    ax1.legend(loc='upper left')

    # Lưu biểu đồ thành file
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot results from CSV')
    parser.add_argument('--csv_file', type=str, default="result/test_qg_log.csv", help='Path to the CSV file containing the results')
    parser.add_argument('--output_file', type=str, default="result/plot.png", help='Path to save the output plot image')
    args = parser.parse_args()

    plot_results(args.csv_file, args.output_file)
