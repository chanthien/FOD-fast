import psutil
import os
# import GPUtil
# import time


class ResourceMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())

    def get_cpu_usage(self):
        # Get the CPU usage percentage
        return self.process.cpu_percent(interval=self.interval)

    def get_memory_usage(self):
        # Get the memory usage in MB
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 ** 2)  # Convert from bytes to MB

    # def get_gpu_usage(self):
    #     # Get the GPU usage percentage and memory usage in MB
    #     gpus = GPUtil.getGPUs()
    #     gpu_usage = []
    #     for gpu in gpus:
    #         gpu_usage.append({
    #             'gpu_id': gpu.id,
    #             'gpu_load': gpu.load * 100,  # Convert to percentage
    #             'gpu_memory_used': gpu.memoryUsed  # Already in MB
    #         })
    #     return gpu_usage

    def print_usage(self):
        print(f"CPU Usage: {self.get_cpu_usage():.2f}%")
        print(f"Memory Usage: {self.get_memory_usage():.2f} MB")

        # gpu_usage = self.get_gpu_usage()
        # if gpu_usage:
        #     for gpu in gpu_usage:
        #         print(f"GPU {gpu['gpu_id']} Usage: {gpu['gpu_load']:.2f}%")
        #         print(f"GPU {gpu['gpu_id']} Memory Usage: {gpu['gpu_memory_used']:.2f} MB")
        # else:
        #     print("No GPU found.")
