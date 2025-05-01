class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_error = float('inf')
        self.no_improve_count = 0
        self.stopped = False

    def __call__(self, epoch, error):
        if error < self.best_error:
            self.best_error = error
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            self.stopped = True
