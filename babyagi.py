#!/usr/bin/env python3
import argparse
import os
import openai
import weaviate
import time
import uuid
import sys
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv
import os

# Parse arguments for optional extensions
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', nargs='+', help='filenames for env')
args = parser.parse_args()

# Load default environment variables (.env)
load_dotenv()

# Set environment variables for optional extensions
if args.env:
    for env_path in args.env:
        load_dotenv(env_path)
        print('Using env from file:', env_path)

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(f"\033[91m\033[1m"+"\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"+"\033[0m\033[0m")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
assert WEAVIATE_URL, "WEAVIATE_URL environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"

#Print OBJECTIVE
print("\033[96m\033[1m"+"\n*****OBJECTIVE*****\n"+"\033[0m\033[0m")
print(OBJECTIVE)

# Configure OpenAI and Weaviate
openai.api_key = OPENAI_API_KEY
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
)

# Create Weaviate index
table_name = YOUR_TABLE_NAME
namespace_uuid = uuid.UUID('2f8ca1d2-ecd1-425d-91a4-41d99d372e0c')

def create_schema(delete_first: bool = False) -> None:
    if delete_first:
        client.schema.delete_class(table_name)
    client.schema.get()
    schema = {
        "classes": [
            {
                "class": table_name,
                "description": "",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text",
                    }
                },
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": "result",
                    },
                    {
                        "dataType": ["text"],
                        "description": "",
                        "name": "task",
                    },
                ],
            },
        ]
    }
    try:
        client.schema.create(schema)
    except weaviate.exceptions.UnexpectedStatusCodeException:
        if delete_first:
            raise

create_schema(delete_first=True)


# Task list
task_list = deque([])

def add_task(task: Dict):
    task_list.append(task)

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def openai_call(prompt: str, model: str = OPENAI_API_MODEL, temperature: float = 0.5, max_tokens: int = 100):
    if not model.startswith('gpt-'):
        # Use completion API
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip()
    else:
        # Use chat completion API
        messages=[{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()

def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]

def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})

def execution_agent(objective: str, task: str) -> str:
    context=context_agent(query=objective, n=5)
    #print("\n*******RELEVANT CONTEXT******\n")
    #print(context)
    prompt =f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, temperature=0.7, max_tokens=2000)

def context_agent(query: str, n: int):
    content = {"concepts": [query]}
    query_obj = client.query.get(table_name, ['result', 'task'])
    result = query_obj.with_near_text(content).with_limit(n).do()
    results = []
    for res in result.get("data", {}).get("Get", {}).get(table_name, []):
        results.append(res['task'])
    #print("***** RESULTS *****")
    #print(results)
    return results

# Add the first task
first_task = {
    "task_id": 1,
    "task_name": YOUR_FIRST_TASK
}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m"+"\n*****TASK LIST*****\n"+"\033[0m\033[0m")
        for t in task_list:
            print(str(t['task_id'])+": "+t['task_name'])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m"+"\n*****NEXT TASK*****\n"+"\033[0m\033[0m")
        print(str(task['task_id'])+": "+task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE,task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m"+"\n*****TASK RESULT*****\n"+"\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Weaviate
        enriched_result = {'data': result}  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']  # extract the actual result from the dictionary

        result_uuid = uuid.uuid5(namespace_uuid, f'Result:{result_id}')
        with client.batch as batch:
            batch.add_data_object(
                dict(result=result, task=task['task_name']),
                table_name,
                result_uuid
            )

    # Step 3: Create new tasks and reprioritize task list
    new_tasks = task_creation_agent(OBJECTIVE,enriched_result, task["task_name"], [t["task_name"] for t in task_list])

    for new_task in new_tasks:
        task_id_counter += 1
        new_task.update({"task_id": task_id_counter})
        add_task(new_task)
    prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again
