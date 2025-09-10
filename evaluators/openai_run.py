import asyncio
import aiohttp
import os
from openai import AsyncOpenAI
from typing import List
from jinja2 import Template
from pydantic import BaseModel, Field
from openai.lib._parsing import type_to_response_format_param
import json
from settings import load_settings
from utils.path import get_assets_dir

MAX_CONCURRENT_REQUESTS = 10

class Answer(BaseModel):
    answer: int = Field(description="The answer to the question. 0 if a, 1 if b, 2 if c, 3 if d, 4 if e, and 5 if a clear answer is not found.")
    
async def call_openai_api(**kwargs):
    settings = load_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        **kwargs
    )
    return response

async def retry_with_delay(func, max_retries=3, delay=10, **kwargs):
    for attempt in range(max_retries):
        try:
            return await func(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

async def process_row(response: dict) -> str:
    template_path = os.path.join(get_assets_dir(), 'eval_prompt.jinja2')

    with open(template_path, 'r') as file:
        eval_template = Template(file.read())
    prompt = eval_template.render(response)
    gen_params = {
            'model': 'gpt-4o-mini-2024-07-18',
            'temperature': 0.0,
            'messages': [{"role": "user", "content": prompt}],
            'response_format': type_to_response_format_param(Answer),
            'stream': False,
            'seed': 42,
        }
    response = await retry_with_delay(call_openai_api, **gen_params)
    return json.loads(response.choices[0].message.content)['answer']

async def process_answers(responses: List[str]) -> List[str]:
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def bounded_process_row(row):
        async with sem:
            return await process_row(row)
    
    for response in responses:
        task = asyncio.ensure_future(bounded_process_row({"RESPONSE": response}))
        tasks.append(task)
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10000)
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results