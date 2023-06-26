import psutil
import subprocess
import pytest


scripts = ['src/train.py', 'src/data_preprocess.py']

@pytest.mark.parametrize("script_path", scripts)
def test_ram_cpu(script_path):

    args = ['--params', 'params.yaml']
    # Start the subprocess running the script
    process = subprocess.Popen(['python', script_path] + args)

    mem_info = psutil.virtual_memory()
    ram_available = mem_info.available

    max_ram_usage, max_cpu_percent = 0, 0
    while process.poll() is None:
        process_info = psutil.Process(process.pid)

        # Get the resident set size (RAM usage) in bytes
        ram_info = process_info.memory_info()
        ram_usage = ram_info.rss
        max_ram_usage = max(max_ram_usage, ram_usage)

        # Get CPU usage percentage
        cpu_percent = psutil.cpu_percent(interval=1)  
        max_cpu_percent = max(max_cpu_percent, cpu_percent)

    max_ram_percent = round(max_ram_usage/ram_available*10, 2)

    assert max_cpu_percent/100 < 0.9, "CPU usage is high, more than 90%"
    assert max_ram_percent < 0.9, "RAM usage is high, more than 90%"
