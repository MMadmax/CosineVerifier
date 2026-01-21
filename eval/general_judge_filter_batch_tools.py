from email import parser
import json
import os
import time
import threading
import queue
import argparse
import asyncio
import ray

import re
import httpx
import importlib
import yaml
from openai import AsyncOpenAI
from tqdm import tqdm

from collections import  defaultdict
from utils.val_prompts import (
                                COSINE_VERIFIER_PROMPT_LABEL, 
                                COMPASS_PROMPT, 
                                COMPASS_COT_PROMPT,
                                XVERIFY_PROMPT,
                                COSINE_VERIFIER_PROMPT_TOOL
                                )

from utils.tools.schemas import OpenAIFunctionToolSchema
import tiktoken

def load_tools_from_config(config_path):
    tool_objects = {}
    tool_schemas = []
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["tools"]

    for tool_config in config:
        try:
            class_path = tool_config["class_name"]
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)

            instance_config = tool_config.get("config", {})
            instance_schema = tool_config.get("tool_schema", {})
            instance_schema = OpenAIFunctionToolSchema.model_validate(instance_schema)
            tool_schemas.append(instance_schema)

            tool_instance = tool_class(config=instance_config, tool_schema=instance_schema)
            tool_name = instance_schema.function.name
            if tool_name:
                tool_objects[tool_name] = tool_instance
        except Exception as e:
            print(f"Error loading tool from config: {tool_config}. Error: {e}")

    return tool_objects, tool_schemas


async def execute_tool_call(tool_call, tool_objects):
    tool_name = tool_call.function.name
    tool_to_call = tool_objects.get(tool_name)
    if not tool_to_call:
        return f"Error: Tool '{tool_name}' not found."

    try:
        tool_args = json.loads(tool_call.function.arguments)
        instance_id = await tool_to_call.create()
        observation, _, _ = await tool_to_call.execute(instance_id, tool_args)
        await tool_to_call.release(instance_id)
        return observation
    except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        return f"Error: Failed to execute tool '{tool_name}' with args {tool_call.function.arguments}. Reason: {e}"


def repeated_dataset(dataset_path, K, repeated_path):
    ds = open_jsonl_folder(dataset_path)
    with open(repeated_path + ".jsonl", "w", encoding="utf-8") as of:
        for i, x in enumerate(ds):
            x["verify_tag_unique_id"] = f"{dataset_path}_{i}"
            of.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_response(repeated_path, repeated_response_path, service_ip_file, K, model_name, verify_type):
    INPUT_FILE = repeated_path + ".jsonl"
    OUTPUT_DIR = repeated_response_path
    MAX_RETRIES = 2
    THREAD_MULTIPLIER = 48
    REFRESH_INTERVAL = 60
    TOOL_CONFIG_PATH = "./utils/tools/sandbox_util_integrate_config.yaml"

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tool_objects, tool_schemas = load_tools_from_config(TOOL_CONFIG_PATH)
    if not tool_objects:
        print("Warning: No tools were loaded. Continuing without tool capabilities.")

    service_writers = {}
    clients_lock = threading.Lock()
    stop_event = threading.Event()
    global_pbar = None
    pbar_lock = threading.Lock()
    tool_usage_counts = defaultdict(int)
    tool_counts_lock = threading.Lock()

    def ensure_output_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_service_ips(ip_file):
        with open(ip_file, "r") as f:
            return {line.strip() for line in f if len(line.strip()) > 4}

    def create_client(ip):
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
            timeout=3600,
        )
        return AsyncOpenAI(
            base_url=f"http://{ip}:30000/v1",
            api_key="NOKEY",
            http_client=http_client,
        )

    def writer_thread_func(ip, result_queue, output_file):
        with open(output_file, "a", encoding="utf-8") as wf:
            while not stop_event.is_set() or not result_queue.empty():
                try:
                    result = result_queue.get(timeout=1)
                    wf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    wf.flush()
                    if global_pbar:
                        with pbar_lock:
                            global_pbar.update(1)
                except queue.Empty:
                    continue

    def start_writer_for_ip(ip):
        q = queue.Queue()
        output_file = os.path.join(OUTPUT_DIR, f"output_{ip.replace('.', '_')}.jsonl")
        thread = threading.Thread(target=writer_thread_func, args=(ip, q, output_file), daemon=True)
        thread.start()
        service_writers[ip] = {"queue": q, "thread": thread}

    async def batch_request(prompt, client, enalbe_thinking=False, tools=None, tool_objects=None, max_turns=5):
        messages = [{"role": "user", "content": prompt.strip()}]

        for _ in range(max_turns):
            for attempt in range(MAX_RETRIES):
                try:
                    request_params = {
                        "messages": messages,
                        "model": "deepseek",
                        "temperature": 0.6,
                        "top_p": 0.85,
                        "max_tokens": 32768,
                        "stream": False,
                        "timeout": 3600,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": enalbe_thinking}},
                    }

                    if tools:
                        request_params["tools"] = [t.model_dump() for t in tools]
                        request_params["tool_choice"] = "auto"

                    response = await client.chat.completions.create(**request_params)
                    response_message = response.choices[0].message

                    if response_message.tool_calls:
                        messages.append(response_message)
                        tool_calls = response_message.tool_calls

                        with tool_counts_lock:
                            for tool_call in tool_calls:
                                tool_usage_counts[tool_call.function.name] += 1

                        tool_responses = await asyncio.gather(
                            *[execute_tool_call(tc, tool_objects) for tc in tool_calls]
                        )

                        for tool_call, observation in zip(tool_calls, tool_responses):
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_call.function.name,
                                    "content": str(observation),
                                }
                            )
                        break
                    else:
                        messages.append(response_message)
                        return response_message.content, messages
                except Exception as e:
                    print(f"[Retry {attempt+1}/{MAX_RETRIES}] Error processing prompt: {e}")
            else:
                return "", messages

        return "Error: Exceeded maximum tool iterations.", messages

    def service_worker_wrapper(ip, task_queue, writer_queue, tool_objects, tool_schemas, verify_type):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(service_worker(ip, task_queue, writer_queue, tool_objects, tool_schemas, verify_type))
        finally:
            loop.close()

    async def service_worker(ip, task_queue, writer_queue, tool_objects, tool_schemas, verify_type):
        client = create_client(ip)
        try:
            while not stop_event.is_set():
                sample = None
                try:
                    sample = task_queue.get(timeout=1)
                    try:
                        result = sample.copy()
                        answer_key = next((k for k in sample if "_ans" in k and "verify" not in k), None)
                        if not answer_key:
                            continue

                        answer = extract_thinking(sample.get(answer_key, ""))
                        if len(answer) > 8192:
                            answer = answer[-8192:]

                        prompt_map = {
                            "STEM": COSINE_VERIFIER_PROMPT_TOOL,
                            "COMPASS": COMPASS_PROMPT,
                            "COMPASS_COT": COMPASS_COT_PROMPT,
                            "XVERIFY": XVERIFY_PROMPT,
                        }
                        prompt_template = prompt_map.get(verify_type, COSINE_VERIFIER_PROMPT_LABEL)
                        prompt = prompt_template.format(
                            question=sample["question"], pred=answer, reference=sample["answer"]
                        )

                        model_response, full_conversation = await batch_request(
                            prompt, client, tools=tool_schemas, tool_objects=tool_objects
                        )

                        serializable_conversation = [
                            msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in full_conversation
                        ]
                        reasoning_trace = format_conversation_to_trace(serializable_conversation)
                        result[f"{model_name}_full_response"] = reasoning_trace
                        result[f"{model_name}_full_message"] = str(full_conversation)
                        result[f"{model_name}_verify_ans"] = model_response
                        writer_queue.put(result)
                    except Exception as e:
                        uid = sample.get("verify_tag_unique_id", "N/A") if sample else "N/A"
                        print(f"Error processing sample {uid}: {e}")
                    finally:
                        if sample is not None:
                            task_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            await client.close()

    def refresh_services(task_queue, all_worker_threads, service_ip_file, verify_type):
        known_ips = set()
        while not stop_event.is_set():
            try:
                current_ips = load_service_ips(service_ip_file)
                with clients_lock:
                    if known_ips != current_ips:
                        for ip in current_ips:
                            if ip not in service_writers:
                                start_writer_for_ip(ip)
                                writer_queue = service_writers[ip]["queue"]
                                for _ in range(THREAD_MULTIPLIER):
                                    worker = threading.Thread(
                                        target=service_worker_wrapper,
                                        args=(ip, task_queue, writer_queue, tool_objects, tool_schemas, verify_type),
                                        daemon=True,
                                    )
                                    worker.start()
                                    all_worker_threads.append(worker)
                        known_ips = current_ips
                time.sleep(REFRESH_INTERVAL)
            except Exception as e:
                print(f"Error refreshing services: {e}")
                time.sleep(REFRESH_INTERVAL)

    def load_processed_sample_ids():
        processed_samples_num = defaultdict(int)
        if not os.path.exists(OUTPUT_DIR):
            return processed_samples_num
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith(".jsonl"):
                with open(os.path.join(OUTPUT_DIR, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get(f"{model_name}_verify_ans", "") == "":
                                continue
                            processed_samples_num[data["verify_tag_unique_id"]] += 1
                        except Exception:
                            continue
        return processed_samples_num

    ensure_output_dir(OUTPUT_DIR)
    processed_samples_num = load_processed_sample_ids()

    samples_to_process = []
    with open(INPUT_FILE, "r", encoding="utf-8") as rf:
        all_lines = rf.readlines()
        for sample_data in tqdm((json.loads(line) for line in all_lines), total=len(all_lines)):
            uid = sample_data.get("verify_tag_unique_id")
            if not uid:
                continue
            processed_num = processed_samples_num.get(uid, 0)
            if processed_num < K:
                for _ in range(K - processed_num):
                    samples_to_process.append(sample_data)

    total_samples = len(samples_to_process)
    if total_samples == 0:
        return

    global_pbar = tqdm(total=total_samples, desc="Processing Samples")
    task_queue = queue.Queue()
    for sample in samples_to_process:
        task_queue.put(sample)

    all_worker_threads = []
    refresh_thread = threading.Thread(
        target=refresh_services, args=(task_queue, all_worker_threads, service_ip_file, verify_type), daemon=True
    )
    refresh_thread.start()

    while not service_writers:
        time.sleep(2)

    try:
        task_queue.join()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop_event.set()
        for worker in all_worker_threads:
            worker.join(timeout=2)
        for ip in service_writers:
            service_writers[ip]["thread"].join(timeout=5)
        global_pbar.close()

        if ray.is_initialized():
            ray.shutdown()

        for func, freq in tool_usage_counts.items():
            print(f"function {func} freq: {freq}")


def calculate_metric(repeated_response_path, model_name, K, verify_type):
    data = open_jsonl_folder(repeated_response_path)

    total = 0
    short_answer_len = 0
    multi_choice_len = 0

    acc = 0
    short_answer_acc = 0
    multi_choice_acc = 0

    true_positives = 0
    actual_positives = 0
    all_tokens = 0

    encoding = tiktoken.get_encoding("cl100k_base")

    for sample in data:
        verify_key = get_verikey(sample)
        if not verify_key:
            continue

        verify_ans = sample.get(verify_key)
        if verify_ans is None:
            continue

        all_tokens += len(encoding.encode(verify_ans))
        verify_ans = extract_thinking(verify_ans)
        gold_ans = sample.get("gold_ans")

        try:
            if verify_type == "STEM":
                verify_ans = find_last_correction(verify_ans)
            elif verify_type == "COMPASS":
                judge = compass_process_judgment(verify_ans)
                if judge == "A":
                    verify_ans = "[Correct]"
                else:
                    verify_ans = "[Incorrect]"

            if gold_ans == "[Correct]":
                actual_positives += 1

            if verify_ans == gold_ans:
                acc += 1
                if sample.get("question_type") == "short_answer":
                    short_answer_acc += 1
                else:
                    multi_choice_acc += 1

                if gold_ans == "[Correct]":
                    true_positives += 1

            total += 1
            if sample.get("question_type") == "short_answer":
                short_answer_len += 1
            else:
                multi_choice_len += 1

        except Exception as e:
            print(verify_ans)
            print(f"{e}, Skipping")
            continue

    total_acc = acc / total if total > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    precision = acc / total if total > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_tokens = all_tokens / len(data) if len(data) > 0 else 0
    sa_acc = short_answer_acc / short_answer_len if short_answer_len > 0 else 0
    mc_acc = multi_choice_acc / multi_choice_len if multi_choice_len > 0 else 0

    print(f"Total accuracy mean@{K} of {total} valid samples: {total_acc}\n")
    print(f"Recall score: {recall}")
    print(f"f1_score: {f1_score}")
    print(f"Average_token: {avg_tokens}")
    print(f"Short answer accuracy mean@{K} {short_answer_len} valid samples: {sa_acc}\n")
    print(f"Multiple choice accuracy mean@{K} {multi_choice_len} valid samples: {mc_acc}\n")


def format_conversation_to_trace(conversation):
    parts = []
    trace_messages = conversation[1:] if len(conversation) > 1 else []

    for msg in trace_messages:
        role = msg.get("role")

        if role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if content:
                parts.append(f"assistant\n{content}")

            if tool_calls:
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        args = tc["function"]["arguments"]

                    call_data = {"name": tc["function"]["name"], "arguments": args}
                    parts.append(f"<tool_call>\n{json.dumps(call_data)}\n</tool_call>")

        elif role == "tool":
            parts.append("user")
            content = msg.get("content")
            parts.append(f"<tool_response>\n{content}\n</tool_response>")

    return "\n".join(parts)


def extract_thinking(solution_str):
    if not isinstance(solution_str, str):
        return solution_str
    if "</think>" in solution_str:
        idx = solution_str.find("</think>")
        return solution_str[idx + len("</think>") :]
    return solution_str


def compass_process_judgment(judgment_str):
    boxed_matches = re.findall(r"boxed{([A-C])}", judgment_str)
    if boxed_matches:
        return boxed_matches[-1]

    if judgment_str in ["A", "B", "C"]:
        return judgment_str

    final_judgment_str = judgment_str.split("Final Judgment:")[-1]
    matches = re.findall(r"\(([A-C])\)*", final_judgment_str)
    if matches:
        return matches[-1]
    matches = re.findall(r"([A-C])", final_judgment_str)
    if matches:
        return matches[-1]
    return ""


def find_last_correction(text):
    if not isinstance(text, str):
        return None

    pos_correct = text.rfind("Correct")
    pos_incorrect = text.rfind("Incorrect")
    last_pos = max(pos_correct, pos_incorrect)

    if last_pos == -1:
        return None
    if last_pos == pos_correct:
        return "[Correct]"
    return "[Incorrect]"


def open_jsonl_file(file_path, mode="r"):
    datas = []
    with open(file_path, mode, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    datas.append(json.loads(line))
                except Exception:
                    continue
    print(f"successfully read {len(datas)} items")
    return datas


def save_jsonl_file(file_path, all_data, mode="w"):
    with open(file_path, mode, encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"successfully completed in {file_path} with {len(all_data)} items")


def open_jsonl_folder(folder_path, mode="r"):
    results = []
    if not os.path.isdir(folder_path):
        print(f"Error: Path is not a directory: {folder_path}")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            results.extend(open_jsonl_file(os.path.join(folder_path, file), mode=mode))

    print(f"load {len(results)} from folder")
    return results


def get_verikey(data):
    for key in data.keys():
        if "_ans" in key and "verify" in key:
            return key
    return ""


def get_anskey(data):
    for key in data.keys():
        if "_ans" in key and "verify" not in key:
            return key
    return ""


def main(args):
    K = args.K
    verify_type = args.verify_type

    for dataset_path in args.dataset_paths:
        for service_ip_file in args.service_ip_files:
            model_name = os.path.splitext(os.path.basename(service_ip_file))[0]
            dataset_name = os.path.basename(dataset_path.rstrip("/"))

            repeated_path_base = os.path.join(args.output_base_dir, f"{model_name}_{dataset_name}")
            repeated_response_dir = f"{repeated_path_base}_output"

            repeated_dataset(dataset_path, K, repeated_path_base)

            get_response(
                repeated_path_base,
                repeated_response_dir,
                service_ip_file,
                K,
                model_name,
                verify_type,
            )

            calculate_metric(
                repeated_response_dir,
                model_name,
                K,
                verify_type,
            )

    print("All processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch model verification.")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--service_ip_files", type=str, nargs="+", required=True)
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_base_dir", type=str, required=True)
    parser.add_argument(
        "--verify_type",
        type=str,
        default="COMPASS",
        choices=["STEM", "COMPASS", "COMPASS_COT", "XVERIFY"],
    )
    args = parser.parse_args()
    main(args)