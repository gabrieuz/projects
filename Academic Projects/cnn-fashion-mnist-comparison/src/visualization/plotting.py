def plot_training_history(histories):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for model_name, hist in histories.items():
        plt.plot(hist["train_accs"], marker='o', label=f'{model_name} Train')
        plt.plot(hist["val_accs"], marker='x', label=f'{model_name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison - Training vs Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for model_name, hist in histories.items():
        plt.plot(hist["train_losses"], marker='o', label=f'{model_name} Train')
        plt.plot(hist["val_losses"], marker='x', label=f'{model_name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison - Training vs Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_final_results(df_results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.bar(df_results['model'], df_results['test_acc'])
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(df_results['model'], df_results['train_time (s)'])
    plt.xlabel('Models')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(df_results['model'], df_results['num_params'])
    plt.xlabel('Models')
    plt.ylabel('Number of Parameters')
    plt.title('Parameter Comparison')
    plt.tight_layout()
    plt.show()