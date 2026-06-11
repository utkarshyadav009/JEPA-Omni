import torch

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {prop.total_memory / 1024**3:.2f} GiB")
            print(f"  Processor Count: {prop.multi_processor_count}")

if __name__ == "__main__":
    main()
