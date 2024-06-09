import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file: str, output_file: str) -> None:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_file)

    # Tạo figure và axis
    fig, ax1 = plt.subplots()

    # Thiết lập vị trí các cột và độ rộng của các cột
    x = range(len(df))
    width = 0.4

    # Tạo biểu đồ cột cho loss
    ax1.bar([i - width/2 for i in x], df['loss'], width=width, label='Loss', color='blue')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Tạo axis thứ hai cho accuracy
    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], df['accuracy'], width=width, label='Accuracy', color='orange')
    ax2.set_ylabel('Accuracy', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Thiết lập tên trục x
    plt.xticks(x, df['model_type'], rotation=45, ha='right')

    # Thiết lập tiêu đề
    plt.title('Loss and Accuracy of Different Models')

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
