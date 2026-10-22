import json
import matplotlib.pyplot as plt
import os
import glob

def plot_history(json_path, save_dir, dataset_name):
    with open(json_path, 'r') as f:
        history = json.load(f)
    
    hist = history['history']
    train_hist = hist.get('train', {})
    val_hist = hist.get('val', {})
    
    if 'loss' not in train_hist:
        return
        
    epochs = range(1, len(train_hist['loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_hist['loss'], label='Train Loss')
    if 'loss' in val_hist:
        plt.plot(epochs, val_hist['loss'], label='Val Loss')
    plt.title(f'{dataset_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_loss.png'))
    plt.close()
    
    # Plot Accuracy
    if 'accuracy' in train_hist:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_hist['accuracy'], label='Train Accuracy')
        if 'accuracy' in val_hist:
            plt.plot(epochs, val_hist['accuracy'], label='Val Accuracy')
        plt.title(f'{dataset_name} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_accuracy.png'))
        plt.close()

def main():
    base_dir = '/home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/research-log-master-lung/own-work/code/master-research-codes/20260614-ips-project/20260614-ips-project/outputs/ips'
    thesis_fig_dir = '/home/axel/Downloads/udsbooks/M2/Thesis/latex-sources/brainstorm/paper-code/research-log-master-lung/own-work/master2-thesis/figures/results'
    os.makedirs(thesis_fig_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(base_dir, '*/*_history.json'))
    for jf in json_files:
        dataset = os.path.basename(os.path.dirname(jf))
        plot_history(jf, thesis_fig_dir, dataset.upper())
        print(f"Generated plots for {dataset.upper()}")

if __name__ == '__main__':
    main()
