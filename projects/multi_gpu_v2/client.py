import base64
import requests
import os
from loguru import logger
import asyncio
import aiohttp

async def mineru_parse_async(session, file_path, server_url='http://127.0.0.1:8000/predict', **options):
    """
    Asynchronous version of the parse function.
    """
    try:
        # Asynchronously read and encode the file
        with open(file_path, 'rb') as f:
            file_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            'file': file_b64,
            'options': options
        }

        # Use the aiohttp session to send the request
        async with session.post(server_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"✅ Processed: {file_path} -> {result.get('output_dir', 'N/A')}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"❌ Server error for {file_path}: {error_text}")
                return {'error': error_text}

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        return {'error': str(e)}


async def main():
    """
    Main function to run all parsing tasks concurrently.
    """
    test_files = [
        '../../demo/pdfs/demo1.pdf',
        '../../demo/pdfs/demo2.pdf',
        '../../demo/pdfs/demo3.pdf',
        '../../demo/pdfs/small_ocr.pdf',
    ]

    test_files = [os.path.join(os.path.dirname(__file__), f) for f in test_files]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    if not existing_files:
        logger.warning("No test files found.")
        return

    # Create an aiohttp session to be reused across requests
    async with aiohttp.ClientSession() as session:
        # === Basic Processing ===
        basic_tasks = [mineru_parse_async(session, file_path) for file_path in existing_files[:2]]

        # === Custom Options ===
        custom_options = {
            'backend': 'pipeline', 'lang': 'ch', 'method': 'auto',
            'formula_enable': True, 'table_enable': True
        }
        # 'backend': 'sglang-engine' requires 24+ GB VRAM per worker

        custom_tasks = [mineru_parse_async(session, file_path, **custom_options) for file_path in existing_files[2:]]

        # Start all tasks
        all_tasks = basic_tasks + custom_tasks

        all_results = await asyncio.gather(*all_tasks)

        logger.info(f"All Results: {all_results}")

        
    logger.info("🎉 All processing completed!")

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())