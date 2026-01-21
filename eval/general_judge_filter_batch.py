import json
import os
import time
import threading
import queue
import argparse
import re
from collections import defaultdict

import httpx
import tiktoken
from openai import OpenAI
from tqdm import tqdm

from utils.val_prompts import (
    COSINE_VERIFIER_PROMPT_LABEL,
    COSINE_VERIFIER_PROMPT_TOOL,
    COMPASS_PROMPT,
    COMPASS_COT_PROMPT,
    XVERIFY_PROMPT,
)


def repeated_dataset(dataset_path, K, repeated_path, ds_keyword="train"):
    ds = open_jsonl_folder(dataset_path)
    os.makedirs(os.path.dirname(repeated_path), exist_ok=True)
    with open(repeated_path + ".jsonl", "w", encoding="utf-8") as of:
        for i, x in enumerate(ds):
            x["verify_tag_unique_id"] = f"{dataset_path}_{i}"
            of.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_response(
    repeated_path,
    repeated_response_path,
    service_ip_file,
    repeated_num=8,
    model_name="model",
    verify_type="COMPASS",
):
    print("------------------start generate------------------")
    SERVICE_IP_FILE = service_ip_file
    INPUT_FILE = repeated_path + ".jsonl"
    OUTPUT_DIR = repeated_response_path
    MAX_RETRIES = 2
    THREAD_MULTIPLIER = 1024
    REFRESH_INTERVAL = 60

    service_clients = {}
    service_writers = {}
    clients_lock = threading.Lock()
    stop_event = threading.Event()
    global_pbar = None

    def ensure_output_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def load_service_ips(ip_file):
        with open(ip_file, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if len(line.strip()) > 4}

    def create_client(ip):
        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
            timeout=3600,
        )
        return OpenAI(
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
                    global_pbar.update(1)
                except queue.Empty:
                    continue

    def start_writer_for_ip(ip):
        q = queue.Queue()
        output_file = os.path.join(OUTPUT_DIR, f"output_{ip.replace('.', '_')}.jsonl")
        thread = threading.Thread(
            target=writer_thread_func, args=(ip, q, output_file), daemon=True
        )
        thread.start()
        service_writers[ip] = {"queue": q, "thread": thread}
        print(f"Started writer for service {ip} writing to {output_file}")

    def refresh_services(task_queue, all_worker_threads):
        known_ips = set()
        while not stop_event.is_set():
            try:
                current_ips = load_service_ips(SERVICE_IP_FILE)
                with clients_lock:
                    if known_ips != current_ips:
                        print(f"Service IPs changed. Old: {known_ips}, New: {current_ips}")
                        for ip in current_ips:
                            if ip not in service_clients:
                                client = create_client(ip)
                                service_clients[ip] = client
                                start_writer_for_ip(ip)
                                writer_queue = service_writers[ip]["queue"]
                                for _ in range(THREAD_MULTIPLIER):
                                    worker = threading.Thread(
                                        target=service_worker,
                                        args=(client, task_queue, writer_queue),
                                        daemon=True,
                                    )
                                    worker.start()
                                    all_worker_threads.append(worker)
                                print(f"Added new service and {THREAD_MULTIPLIER} workers for: {ip}")
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

    def batch_request(prompt, client, enalbe_thinking=True):
        messages = [{"role": "user", "content": prompt.strip()}]
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model="deepseek",
                    temperature=0.0,
                    top_p=0.85,
                    max_tokens=8192,
                    stream=False,
                    timeout=3600,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enalbe_thinking}},
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"[Retry {attempt+1}/{MAX_RETRIES}] Error processing prompt: {e}")
        return ""

    def service_worker(client, task_queue, writer_queue):
        while not stop_event.is_set():
            try:
                sample = task_queue.get(timeout=1)
                result = sample.copy()
                answer_key = ""
                for key in sample.keys():
                    if "_ans" in key and "verify" not in key:
                        answer_key = key

                try:
                    answer = sample[answer_key]
                except Exception as e:
                    print(f"Error during service: {e}")
                    answer = "INVALID"

                answer = extract_thinking(answer)
                if len(answer) > 16384:
                    answer = answer[-16384:]

                if verify_type == "STEM":
                    prompt = COSINE_VERIFIER_PROMPT_LABEL.format(
                        question=sample["question"], pred=answer, reference=sample["answer"]
                    )
                elif verify_type == "COMPASS":
                    prompt = COMPASS_PROMPT.format(
                        question=sample["question"], pred=answer, reference=sample["answer"]
                    )
                elif verify_type == "COMPASS_COT":
                    prompt = COMPASS_COT_PROMPT.format(
                        question=sample["question"], pred=answer, reference=sample["answer"]
                    )
                elif verify_type == "XVERIFY":
                    prompt = XVERIFY_PROMPT.format(
                        question=sample["question"], pred=answer, reference=sample["answer"]
                    )
                else:
                    prompt = COSINE_VERIFIER_PROMPT_LABEL.format(
                        question=sample["question"], pred=answer, reference=sample["answer"]
                    )

                model_response = batch_request(prompt, client)
                result[f"{model_name}_verify_ans"] = model_response
                writer_queue.put(result)
                task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in service worker: {e}")
                try:
                    task_queue.task_done()
                except Exception:
                    pass

    ensure_output_dir(OUTPUT_DIR)

    processed_samples_num = load_processed_sample_ids()
    print(f"Found {len(processed_samples_num)} questions with at least one processed response.")

    samples_to_process = []
    with open(INPUT_FILE, "r", encoding="utf-8") as rf:
        all_lines = rf.readlines()
        for sample_data in tqdm((json.loads(line) for line in all_lines), total=len(all_lines)):
            uid = sample_data.get("verify_tag_unique_id")
            if not uid:
                continue
            processed_num = processed_samples_num.get(uid, 0)
            if processed_num < repeated_num:
                for _ in range(repeated_num - processed_num):
                    samples_to_process.append(sample_data)

    total_samples = len(samples_to_process)
    print(f"Total new samples to process: {total_samples}")
    if total_samples == 0:
        print("No new samples to process. Exiting.")
        return

    global_pbar = tqdm(total=total_samples, desc="Processing Samples")

    task_queue = queue.Queue()
    for sample in samples_to_process:
        task_queue.put(sample)

    all_worker_threads = []
    refresh_thread = threading.Thread(
        target=refresh_services, args=(task_queue, all_worker_threads), daemon=True
    )
    refresh_thread.start()

    print("Waiting for initial services to come online...")
    while not service_clients:
        time.sleep(2)
    print(f"Initial services loaded: {list(service_clients.keys())}")

    try:
        task_queue.join()
        print("All tasks have been processed.")
    except (KeyboardInterrupt, SystemExit):
        print("Interruption received. Shutting down...")
    finally:
        stop_event.set()
        for worker in all_worker_threads:
            worker.join(timeout=2)
        for ip in service_writers:
            service_writers[ip]["thread"].join(timeout=5)
        global_pbar.close()
        print("All processes finished.")


def calculate_metric(repeated_response_path, model_name="model", K=1, verify_type="COMPASS"):
    data = open_jsonl_folder(repeated_response_path)
    if not data:
        print(f"No data found in {repeated_response_path}. Skipping metrics.")
        return

    total = 0
    short_answer_len = 0
    multi_choice_len = 0

    acc = 0
    short_answer_acc = 0
    multi_choice_acc = 0

    true_positives = 0
    actual_positives = 0
    encoding = tiktoken.get_encoding("cl100k_base")
    all_tokens = 0

    for sample in data:
        verify_ans = sample.get(f"{model_name}_verify_ans")
        if not verify_ans:
            continue

        verify_ans = extract_thinking(verify_ans)
        gold_ans = sample["gold_ans"]
        tokens = encoding.encode(verify_ans)

        try:
            if verify_type == "STEM":
                verify_ans = find_last_correction(verify_ans)
            elif verify_type in ("COMPASS", "COMPASS_COT"):
                judge = compass_process_judgment(verify_ans)
                verify_ans = "[Correct]" if judge == "A" else "[Incorrect]"

            if gold_ans == "[Correct]":
                actual_positives += 1

            if verify_ans == gold_ans:
                acc += 1
                if sample["question_type"] == "short_answer":
                    short_answer_acc += 1
                else:
                    multi_choice_acc += 1
                if gold_ans == "[Correct]":
                    true_positives += 1

            all_tokens += len(tokens)
            total += 1
            if sample["question_type"] == "short_answer":
                short_answer_len += 1
            else:
                multi_choice_len += 1
        except Exception as e:
            print(verify_ans)
            print(f"{e}, Skipping")
            continue

    print(f"\n--- Metrics for {model_name} (k={K}) ---")
    if total > 0:
        precision = acc / total
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        print(f"Total accuracy mean@{K} of {total} valid samples: {precision}\n")
        print(f"Recall score: {recall}")
        print(f"f1_score: {f1_score}")
        print(f"Average_token: {all_tokens / total}")
    else:
        print("No valid samples found to calculate total accuracy.")

    if short_answer_len > 0:
        print(
            f"Short answer accuracy mean@{K} {short_answer_len} valid samples: "
            f"{short_answer_acc / short_answer_len}\n"
        )
    else:
        print("No valid short answer samples found.")

    if multi_choice_len > 0:
        print(
            f"Multiple choice accuracy mean@{K} {multi_choice_len} valid samples: "
            f"{multi_choice_acc / multi_choice_len}\n"
        )
    else:
        print("No valid multiple choice samples found.")
    print("--- End of Metrics ---")


def extract_thinking(solution_str):
    if "</think>" in solution_str:
        idx = solution_str.find("</think>")
        return solution_str[idx + len("</think>") :]
    return solution_str


def compass_process_judgment(judgment_str: str) -> str:
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
    if text is None:
        return None
    pos_correct = text.rfind("Correct")
    pos_incorrect = text.rfind("Incorrect")
    last_pos = max(pos_correct, pos_incorrect)
    if last_pos == -1:
        return None
    return "[Correct]" if last_pos == pos_correct else "[Incorrect]"


def open_jsonl_file(file_path, mode="r"):
    datas = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return []
    with open(file_path, mode, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    datas.append(json.loads(line))
                except Exception:
                    continue
    print(f"successfully read {len(datas)} items from {file_path}")
    return datas


def save_jsonl_file(file_path, all_data, mode="w"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"successfully completed in {file_path} with {len(all_data)} items")


def open_jsonl_folder(folder_path, mode="r"):
    results = []
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder not found {folder_path}")
        return []
    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            results.extend(open_jsonl_file(os.path.join(folder_path, file), mode=mode))
    print(f"load {len(results)} from folder {folder_path}")
    return results


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
            get_response(repeated_path_base, repeated_response_dir, service_ip_file, K, model_name, verify_type)
            calculate_metric(repeated_response_dir, model_name, K, verify_type)

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
