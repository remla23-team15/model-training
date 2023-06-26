import psutil
import subprocess
import pytest


scripts = ['src/train.py', 'src/data_preprocess.py']

@pytest.mark.parametrize("script_path", scripts)
def test_ram_cpu(script_path):

    # Start the subprocess running the script
    process = subprocess.Popen(['python', script_path])

    mem_info = psutil.virtual_memory()
    ram_available = mem_info.available

    max_ram_usage = 0
    max_cpu_percent = 0
    while process.poll() is None:
        process_info = psutil.Process(process.pid)

        # Get the resident set size (RAM usage) in bytes
        ram_info = process_info.memory_info()
        ram_usage = ram_info.rss
        max_ram_usage = max(max_ram_usage, ram_usage)

        # Get CPU usage percentage
        cpu_percent = psutil.cpu_percent(interval=1)  
        max_cpu_percent = max(max_cpu_percent, cpu_percent)


    max_ram_percent = round(max_ram_usage/ram_available*10000, 2)
    print(f'max_cpu: {max_cpu_percent}%')
    print(f'max_ram: {max_ram_percent}%')
    assert max_cpu_percent < 90, "CPU usage is high, more than 90%"
    assert max_ram_percent < 90, "RAM usage is high, more than 90%"
    print("\n\n\nTEST RAM CPU ALL GOOD!!!\n\n\n")
