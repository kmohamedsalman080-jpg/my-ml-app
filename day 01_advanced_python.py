import time

# Custom Exception for Data Processing Errors
class DataProcessingError(Exception):
    pass

# Decorator to measure execution time
def execution_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ [{func.__name__}] Execution Time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Main Class
class DataPreprocessor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @execution_timer
    def clean_data(self, raw_data: list):
        if not raw_data:
            raise DataProcessingError("Dataset Empty-a irukku! Processing panna mudiyadhu.")
        
        # Generator for processing items in batches/streaming
        print(f"Cleaning dataset: {self.dataset_name}...")
        cleaned = [item.strip().lower() for item in raw_data if isinstance(item, str) and item.strip()]
        return cleaned

# --- Execution / Testing ---
if __name__ == "__main__":
    processor = DataPreprocessor("AI_Sentinel_Logs")
    sample_data = ["  MalwareDetected  ", "", None, "  SYSTEM_OK ", "   "]

    try:
        cleaned_logs = processor.clean_data(sample_data)
        print("✅ Cleaned Data Output:", cleaned_logs)
    except DataProcessingError as e:
        print("❌ Error:", e)
        