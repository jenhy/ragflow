#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# from beartype import BeartypeConf
# from beartype.claw import beartype_all  # <-- you didn't sign up for this
# beartype_all(conf=BeartypeConf(violation_type=UserWarning))    # <-- emit warnings from all code
import random
import sys
import threading
import time

from api.utils.api_utils import timeout, is_strong_enough
from api.utils.log_utils import init_root_logger, get_project_base_directory
from graphrag.general.index import run_graphrag
from graphrag.utils import get_llm_cache, set_llm_cache, get_tags_from_cache, set_tags_to_cache
from rag.prompts import keyword_extraction, question_proposal, content_tagging

import logging
import os
from datetime import datetime
import json
import xxhash
import copy
import re
from functools import partial
from io import BytesIO
from multiprocessing.context import TimeoutError
from timeit import default_timer as timer
import tracemalloc
import signal
import trio
import exceptiongroup
import faulthandler

import numpy as np
from peewee import DoesNotExist

from api.db import LLMType, ParserType
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.db.services.task_service import TaskService, has_canceled
from api.db.services.file2document_service import File2DocumentService
from api import settings
from api.versions import get_ragflow_version
from api.db.db_models import close_connection
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, \
    email, tag
from rag.nlp import search, rag_tokenizer
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.settings import DOC_MAXIMUM_SIZE, DOC_BULK_SIZE, EMBEDDING_BATCH_SIZE, SVR_CONSUMER_GROUP_NAME, get_svr_queue_name, get_svr_queue_names, print_rag_settings, TAG_FLD, PAGERANK_FLD
from rag.utils import num_tokens_from_string, truncate
from rag.utils.redis_conn import REDIS_CONN, RedisDistributedLock
from rag.utils.storage_factory import STORAGE_IMPL
from graphrag.utils import chat_limiter

BATCH_SIZE = 64

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: naive,
    ParserType.TAG.value: tag
}

UNACKED_ITERATOR = None

CONSUMER_NO = "0" if len(sys.argv) < 2 else sys.argv[1]
CONSUMER_NAME = "task_executor_" + CONSUMER_NO
BOOT_AT = datetime.now().astimezone().isoformat(timespec="milliseconds")
PENDING_TASKS = 0
LAG_TASKS = 0
DONE_TASKS = 0
FAILED_TASKS = 0

CURRENT_TASKS = {}

MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', "5"))
MAX_CONCURRENT_CHUNK_BUILDERS = int(os.environ.get('MAX_CONCURRENT_CHUNK_BUILDERS', "1"))
MAX_CONCURRENT_MINIO = int(os.environ.get('MAX_CONCURRENT_MINIO', '10'))
task_limiter = trio.Semaphore(MAX_CONCURRENT_TASKS)
chunk_limiter = trio.CapacityLimiter(MAX_CONCURRENT_CHUNK_BUILDERS)
embed_limiter = trio.CapacityLimiter(MAX_CONCURRENT_CHUNK_BUILDERS)
minio_limiter = trio.CapacityLimiter(MAX_CONCURRENT_MINIO)
kg_limiter = trio.CapacityLimiter(2)
WORKER_HEARTBEAT_TIMEOUT = int(os.environ.get('WORKER_HEARTBEAT_TIMEOUT', '120'))
stop_event = threading.Event()


def signal_handler(sig, frame):
    logging.info("Received interrupt signal, shutting down...")
    stop_event.set()
    time.sleep(1)
    sys.exit(0)


# SIGUSR1 handler: start tracemalloc and take snapshot
def start_tracemalloc_and_snapshot(signum, frame):
    if not tracemalloc.is_tracing():
        logging.info("start tracemalloc")
        tracemalloc.start()
    else:
        logging.info("tracemalloc is already running")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = f"snapshot_{timestamp}.trace"
    snapshot_file = os.path.abspath(os.path.join(get_project_base_directory(), "logs", f"{os.getpid()}_snapshot_{timestamp}.trace"))

    snapshot = tracemalloc.take_snapshot()
    snapshot.dump(snapshot_file)
    current, peak = tracemalloc.get_traced_memory()
    if sys.platform == "win32":
        import  psutil
        process = psutil.Process()
        max_rss = process.memory_info().rss / 1024
    else:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info(f"taken snapshot {snapshot_file}. max RSS={max_rss / 1000:.2f} MB, current memory usage: {current / 10**6:.2f} MB, Peak memory usage: {peak / 10**6:.2f} MB")

# SIGUSR2 handler: stop tracemalloc
def stop_tracemalloc(signum, frame):
    if tracemalloc.is_tracing():
        logging.info("stop tracemalloc")
        tracemalloc.stop()
    else:
        logging.info("tracemalloc not running")

class TaskCanceledException(Exception):
    def __init__(self, msg):
        self.msg = msg


def set_progress(task_id, from_page=0, to_page=-1, prog=None, msg="Processing..."):
    try:
        if prog is not None and prog < 0:
            msg = "[ERROR]" + msg
        cancel = has_canceled(task_id)

        if cancel:
            msg += " [Canceled]"
            prog = -1

        if to_page > 0:
            if msg:
                if from_page < to_page:
                    msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
        if msg:
            msg = datetime.now().strftime("%H:%M:%S") + " " + msg
        d = {"progress_msg": msg}
        if prog is not None:
            d["progress"] = prog

        TaskService.update_progress(task_id, d)

        close_connection()
        if cancel:
            raise TaskCanceledException(msg)
        logging.info(f"set_progress({task_id}), progress: {prog}, progress_msg: {msg}")
    except DoesNotExist:
        logging.warning(f"set_progress({task_id}) got exception DoesNotExist")
    except Exception:
        logging.exception(f"set_progress({task_id}), progress: {prog}, progress_msg: {msg}, got exception")


async def collect():
    global CONSUMER_NAME, DONE_TASKS, FAILED_TASKS
    global UNACKED_ITERATOR

    svr_queue_names = get_svr_queue_names()
    try:
        if not UNACKED_ITERATOR:
            UNACKED_ITERATOR = REDIS_CONN.get_unacked_iterator(svr_queue_names, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
        try:
            redis_msg = next(UNACKED_ITERATOR)
        except StopIteration:
            for svr_queue_name in svr_queue_names:
                redis_msg = REDIS_CONN.queue_consumer(svr_queue_name, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
                if redis_msg:
                    break
    except Exception:
        logging.exception("collect got exception")
        return None, None

    if not redis_msg:
        return None, None
    msg = redis_msg.get_message()
    if not msg:
        logging.error(f"collect got empty message of {redis_msg.get_msg_id()}")
        redis_msg.ack()
        return None, None

    canceled = False
    task = TaskService.get_task(msg["id"])
    if task:
        canceled = has_canceled(task["id"])
    if not task or canceled:
        state = "is unknown" if not task else "has been cancelled"
        FAILED_TASKS += 1
        logging.warning(f"collect task {msg['id']} {state}")
        redis_msg.ack()
        return None, None
    task["task_type"] = msg.get("task_type", "")
    return redis_msg, task


async def get_storage_binary(bucket, name):
    return await trio.to_thread.run_sync(lambda: STORAGE_IMPL.get(bucket, name))


@timeout(60*40, 1)
async def build_chunks(task, progress_callback):
    if task["size"] > DOC_MAXIMUM_SIZE:
        set_progress(task["id"], prog=-1, msg="File size exceeds( <= %dMb )" %
                                              (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
        return []

    chunker = FACTORY[task["parser_id"].lower()]
    try:
        st = timer()
        bucket, name = File2DocumentService.get_storage_address(doc_id=task["doc_id"])
        binary = await get_storage_binary(bucket, name)
        logging.info("From minio({}) {}/{}".format(timer() - st, task["location"], task["name"]))
    except TimeoutError:
        progress_callback(-1, "Internal server error: Fetch file from minio timeout. Could you try it again.")
        logging.exception(
            "Minio {}/{} got timeout: Fetch file from minio timeout.".format(task["location"], task["name"]))
        raise
    except Exception as e:
        if re.search("(No such file|not found)", str(e)):
            progress_callback(-1, "Can not find file <%s> from minio. Could you try it again?" % task["name"])
        else:
            progress_callback(-1, "Get file from minio: %s" % str(e).replace("'", ""))
        logging.exception("Chunking {}/{} got exception".format(task["location"], task["name"]))
        raise

    try:
        async with chunk_limiter:
            cks = await trio.to_thread.run_sync(lambda: chunker.chunk(task["name"], binary=binary, from_page=task["from_page"],
                                to_page=task["to_page"], lang=task["language"], callback=progress_callback,
                                kb_id=task["kb_id"], parser_config=task["parser_config"], tenant_id=task["tenant_id"]))
        logging.info("Chunking({}) {}/{} done".format(timer() - st, task["location"], task["name"]))
    except TaskCanceledException:
        raise
    except Exception as e:
        progress_callback(-1, "Internal server error while chunking: %s" % str(e).replace("'", ""))
        logging.exception("Chunking {}/{} got exception".format(task["location"], task["name"]))
        raise

    docs = []
    doc = {
        "doc_id": task["doc_id"],
        "kb_id": str(task["kb_id"])
    }
    if task["pagerank"]:
        doc[PAGERANK_FLD] = int(task["pagerank"])
    st = timer()

    @timeout(60)
    async def upload_to_minio(document, chunk):
        try:
            d = copy.deepcopy(document)
            d.update(chunk)
            d["id"] = xxhash.xxh64((chunk["content_with_weight"] + str(d["doc_id"])).encode("utf-8")).hexdigest()
            d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
            d["create_timestamp_flt"] = datetime.now().timestamp()
            if not d.get("image"):
                _ = d.pop("image", None)
                d["img_id"] = ""
                docs.append(d)
                return

            output_buffer = BytesIO()
            try:
                if isinstance(d["image"], bytes):
                    output_buffer.write(d["image"])
                    output_buffer.seek(0)
                else:
                    # If the image is in RGBA mode, convert it to RGB mode before saving it in JPEG format.
                    if d["image"].mode in ("RGBA", "P"):
                        converted_image = d["image"].convert("RGB")
                        d["image"].close()  # Close original image
                        d["image"] = converted_image
                    d["image"].save(output_buffer, format='JPEG')

                async with minio_limiter:
                    await trio.to_thread.run_sync(lambda: STORAGE_IMPL.put(task["kb_id"], d["id"], output_buffer.getvalue()))
                d["img_id"] = "{}-{}".format(task["kb_id"], d["id"])
                if not isinstance(d["image"], bytes):
                    d["image"].close()
                del d["image"]  # Remove image reference
                docs.append(d)
            finally:
                output_buffer.close()  # Ensure BytesIO is always closed
        except Exception:
            logging.exception(
                "Saving image of chunk {}/{}/{} got exception".format(task["location"], task["name"], d["id"]))
            raise

    async with trio.open_nursery() as nursery:
        for ck in cks:
            nursery.start_soon(upload_to_minio, doc, ck)

    el = timer() - st
    logging.info("MINIO PUT({}) cost {:.3f} s".format(task["name"], el))

    if task["parser_config"].get("auto_keywords", 0):
        st = timer()
        progress_callback(msg="Start to generate keywords for every chunk ...")
        chat_mdl = LLMBundle(task["tenant_id"], LLMType.CHAT, llm_name=task["llm_id"], lang=task["language"])

        async def doc_keyword_extraction(chat_mdl, d, topn):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "keywords", {"topn": topn})
            if not cached:
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(lambda: keyword_extraction(chat_mdl, d["content_with_weight"], topn))
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "keywords", {"topn": topn})
            if cached:
                d["important_kwd"] = cached.split(",")
                d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))
            return
        async with trio.open_nursery() as nursery:
            for d in docs:
                nursery.start_soon(doc_keyword_extraction, chat_mdl, d, task["parser_config"]["auto_keywords"])
        progress_callback(msg="Keywords generation {} chunks completed in {:.2f}s".format(len(docs), timer() - st))

    if task["parser_config"].get("auto_questions", 0):
        st = timer()
        progress_callback(msg="Start to generate questions for every chunk ...")
        chat_mdl = LLMBundle(task["tenant_id"], LLMType.CHAT, llm_name=task["llm_id"], lang=task["language"])

        async def doc_question_proposal(chat_mdl, d, topn):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "question", {"topn": topn})
            if not cached:
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(lambda: question_proposal(chat_mdl, d["content_with_weight"], topn))
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "question", {"topn": topn})
            if cached:
                d["question_kwd"] = cached.split("\n")
                d["question_tks"] = rag_tokenizer.tokenize("\n".join(d["question_kwd"]))
        async with trio.open_nursery() as nursery:
            for d in docs:
                nursery.start_soon(doc_question_proposal, chat_mdl, d, task["parser_config"]["auto_questions"])
        progress_callback(msg="Question generation {} chunks completed in {:.2f}s".format(len(docs), timer() - st))

    if task["kb_parser_config"].get("tag_kb_ids", []):
        progress_callback(msg="Start to tag for every chunk ...")
        kb_ids = task["kb_parser_config"]["tag_kb_ids"]
        tenant_id = task["tenant_id"]
        topn_tags = task["kb_parser_config"].get("topn_tags", 3)
        S = 1000
        st = timer()
        examples = []
        all_tags = get_tags_from_cache(kb_ids)
        if not all_tags:
            all_tags = settings.retrievaler.all_tags_in_portion(tenant_id, kb_ids, S)
            set_tags_to_cache(kb_ids, all_tags)
        else:
            all_tags = json.loads(all_tags)

        chat_mdl = LLMBundle(task["tenant_id"], LLMType.CHAT, llm_name=task["llm_id"], lang=task["language"])

        docs_to_tag = []
        for d in docs:
            task_canceled = has_canceled(task["id"])
            if task_canceled:
                progress_callback(-1, msg="Task has been canceled.")
                return
            if settings.retrievaler.tag_content(tenant_id, kb_ids, d, all_tags, topn_tags=topn_tags, S=S) and len(d[TAG_FLD]) > 0:
                examples.append({"content": d["content_with_weight"], TAG_FLD: d[TAG_FLD]})
            else:
                docs_to_tag.append(d)

        async def doc_content_tagging(chat_mdl, d, topn_tags):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], all_tags, {"topn": topn_tags})
            if not cached:
                picked_examples = random.choices(examples, k=2) if len(examples)>2 else examples
                if not picked_examples:
                    picked_examples.append({"content": "This is an example", TAG_FLD: {'example': 1}})
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(lambda: content_tagging(chat_mdl, d["content_with_weight"], all_tags, picked_examples, topn=topn_tags))
                if cached:
                    cached = json.dumps(cached)
            if cached:
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, all_tags, {"topn": topn_tags})
                d[TAG_FLD] = json.loads(cached)
        async with trio.open_nursery() as nursery:
            for d in docs_to_tag:
                nursery.start_soon(doc_content_tagging, chat_mdl, d, topn_tags)
        progress_callback(msg="Tagging {} chunks completed in {:.2f}s".format(len(docs), timer() - st))

    return docs


def init_kb(row, vector_size: int):
    idxnm = search.index_name(row["tenant_id"])
    return settings.docStoreConn.createIdx(idxnm, row.get("kb_id", ""), vector_size)


@timeout(60*20)
async def embedding(docs, mdl, parser_config=None, callback=None):
    if parser_config is None:
        parser_config = {}
    tts, cnts = [], []
    for d in docs:
        tts.append(d.get("docnm_kwd", "Title"))
        c = "\n".join(d.get("question_kwd", []))
        if not c:
            c = d["content_with_weight"]
        c = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", c)
        if not c:
            c = "None"
        cnts.append(c)

    tk_count = 0
    if len(tts) == len(cnts):
        vts, c = await trio.to_thread.run_sync(lambda: mdl.encode(tts[0: 1]))
        tts = np.concatenate([vts for _ in range(len(tts))], axis=0)
        tk_count += c

    cnts_ = np.array([])
    for i in range(0, len(cnts), EMBEDDING_BATCH_SIZE):
        async with embed_limiter:
            vts, c = await trio.to_thread.run_sync(lambda: mdl.encode([truncate(c, mdl.max_length-10) for c in cnts[i: i + EMBEDDING_BATCH_SIZE]]))
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_
    filename_embd_weight = parser_config.get("filename_embd_weight", 0.1) # due to the db support none value
    if not filename_embd_weight:
        filename_embd_weight = 0.1
    title_w = float(filename_embd_weight)
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    vector_size = 0
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        vector_size = len(v)
        d["q_%d_vec" % len(v)] = v
    return tk_count, vector_size


@timeout(3600)
async def run_raptor(row, chat_mdl, embd_mdl, vector_size, callback=None):
    # Pressure test for GraphRAG task
    await is_strong_enough(chat_mdl, embd_mdl)
    chunks = []
    vctr_nm = "q_%d_vec"%vector_size
    for d in settings.retrievaler.chunk_list(row["doc_id"], row["tenant_id"], [str(row["kb_id"])],
                                             fields=["content_with_weight", vctr_nm]):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    original_length = len(chunks)
    chunks = await raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    if row["pagerank"]:
        doc[PAGERANK_FLD] = int(row["pagerank"])
    res = []
    tk_count = 0
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        d["id"] = xxhash.xxh64((content + str(d["doc_id"])).encode("utf-8")).hexdigest()
        d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    return res, tk_count


@timeout(60*60, 1)
async def do_handle_task(task):
    task_id = task["id"]
    task_from_page = task["from_page"]
    task_to_page = task["to_page"]
    task_tenant_id = task["tenant_id"]
    task_embedding_id = task["embd_id"]
    task_language = task["language"]
    task_llm_id = task["llm_id"]
    task_dataset_id = task["kb_id"]
    task_doc_id = task["doc_id"]
    task_document_name = task["name"]
    task_parser_config = task["parser_config"]
    task_start_ts = timer()

    # prepare the progress callback function
    progress_callback = partial(set_progress, task_id, task_from_page, task_to_page)

    # FIXME: workaround, Infinity doesn't support table parsing method, this check is to notify user
    lower_case_doc_engine = settings.DOC_ENGINE.lower()
    if lower_case_doc_engine == 'infinity' and task['parser_id'].lower() == 'table':
        error_message = "Table parsing method is not supported by Infinity, please use other parsing methods or use Elasticsearch as the document engine."
        progress_callback(-1, msg=error_message)
        raise Exception(error_message)

    task_canceled = has_canceled(task_id)
    if task_canceled:
        progress_callback(-1, msg="Task has been canceled.")
        return

    try:
        # bind embedding model
        embedding_model = LLMBundle(task_tenant_id, LLMType.EMBEDDING, llm_name=task_embedding_id, lang=task_language)
        await is_strong_enough(None, embedding_model)
        vts, _ = embedding_model.encode(["ok"])
        vector_size = len(vts[0])
    except Exception as e:
        error_message = f'Fail to bind embedding model: {str(e)}'
        progress_callback(-1, msg=error_message)
        logging.exception(error_message)
        raise

    init_kb(task, vector_size)

    # Either using RAPTOR or Standard chunking methods
    if task.get("task_type", "") == "raptor":
        # bind LLM for raptor
        chat_model = LLMBundle(task_tenant_id, LLMType.CHAT, llm_name=task_llm_id, lang=task_language)
        await is_strong_enough(chat_model, None)
        # run RAPTOR
        async with kg_limiter:
            chunks, token_count = await run_raptor(task, chat_model, embedding_model, vector_size, progress_callback)
    # Either using graphrag or Standard chunking methods
    elif task.get("task_type", "") == "graphrag":
        if not task_parser_config.get("graphrag", {}).get("use_graphrag", False):
            progress_callback(prog=-1.0, msg="Internal configuration error.")
            return
        graphrag_conf = task["kb_parser_config"].get("graphrag", {})
        start_ts = timer()
        chat_model = LLMBundle(task_tenant_id, LLMType.CHAT, llm_name=task_llm_id, lang=task_language)
        await is_strong_enough(chat_model, None)
        with_resolution = graphrag_conf.get("resolution", False)
        with_community = graphrag_conf.get("community", False)
        async with kg_limiter:
            await run_graphrag(task, task_language, with_resolution, with_community, chat_model, embedding_model, progress_callback)
        progress_callback(prog=1.0, msg="Knowledge Graph done ({:.2f}s)".format(timer() - start_ts))
        return
    else:
        # Standard chunking methods
        start_ts = timer()
        chunks = await build_chunks(task, progress_callback)
        logging.info("Build document {}: {:.2f}s".format(task_document_name, timer() - start_ts))
        if not chunks:
            progress_callback(1., msg=f"No chunk built from {task_document_name}")
            return
        # TODO: exception handler
        ## set_progress(task["did"], -1, "ERROR: ")
        progress_callback(msg="Generate {} chunks".format(len(chunks)))
        start_ts = timer()
        try:
            token_count, vector_size = await embedding(chunks, embedding_model, task_parser_config, progress_callback)
        except Exception as e:
            error_message = "Generate embedding error:{}".format(str(e))
            progress_callback(-1, error_message)
            logging.exception(error_message)
            token_count = 0
            raise
        progress_message = "Embedding chunks ({:.2f}s)".format(timer() - start_ts)
        logging.info(progress_message)
        progress_callback(msg=progress_message)

    chunk_count = len(set([chunk["id"] for chunk in chunks]))
    start_ts = timer()
    doc_store_result = ""

    async def delete_image(kb_id, chunk_id):
        try:
            async with minio_limiter:
                STORAGE_IMPL.delete(kb_id, chunk_id)
        except Exception:
            logging.exception(
                "Deleting image of chunk {}/{}/{} got exception".format(task["location"], task["name"], chunk_id))
            raise

    for b in range(0, len(chunks), DOC_BULK_SIZE):
        doc_store_result = await trio.to_thread.run_sync(lambda: settings.docStoreConn.insert(chunks[b:b + DOC_BULK_SIZE], search.index_name(task_tenant_id), task_dataset_id))
        task_canceled = has_canceled(task_id)
        if task_canceled:
            progress_callback(-1, msg="Task has been canceled.")
            return
        if b % 128 == 0:
            progress_callback(prog=0.8 + 0.1 * (b + 1) / len(chunks), msg="")
        if doc_store_result:
            error_message = f"Insert chunk error: {doc_store_result}, please check log file and Elasticsearch/Infinity status!"
            progress_callback(-1, msg=error_message)
            raise Exception(error_message)
        chunk_ids = [chunk["id"] for chunk in chunks[:b + DOC_BULK_SIZE]]
        chunk_ids_str = " ".join(chunk_ids)
        try:
            TaskService.update_chunk_ids(task["id"], chunk_ids_str)
        except DoesNotExist:
            logging.warning(f"do_handle_task update_chunk_ids failed since task {task['id']} is unknown.")
            doc_store_result = await trio.to_thread.run_sync(lambda: settings.docStoreConn.delete({"id": chunk_ids}, search.index_name(task_tenant_id), task_dataset_id))
            async with trio.open_nursery() as nursery:
                for chunk_id in chunk_ids:
                    nursery.start_soon(delete_image, task_dataset_id, chunk_id)
            progress_callback(-1, msg=f"Chunk updates failed since task {task['id']} is unknown.")
            return

    logging.info("Indexing doc({}), page({}-{}), chunks({}), elapsed: {:.2f}".format(task_document_name, task_from_page,
                                                                                     task_to_page, len(chunks),
                                                                                     timer() - start_ts))

    DocumentService.increment_chunk_num(task_doc_id, task_dataset_id, token_count, chunk_count, 0)

    time_cost = timer() - start_ts
    task_time_cost = timer() - task_start_ts
    progress_callback(prog=1.0, msg="Indexing done ({:.2f}s). Task done ({:.2f}s)".format(time_cost, task_time_cost))
    logging.info(
        "Chunk doc({}), page({}-{}), chunks({}), token({}), elapsed:{:.2f}".format(task_document_name, task_from_page,
                                                                                   task_to_page, len(chunks),
                                                                                   token_count, task_time_cost))


async def handle_task():
    """
    这是一个异步函数，负责获取一个任务并处理它，同时处理成功、失败和无任务的情况。
    handle_task 函数是一个非常重要的控制器。它负责获取任务、处理任务、处理任务的成功和失败状态、并最终向消息队列确认处理完成，从而构成了 RAGFlow 任务执行器中一个完整、健壮、可靠的任务处理生命周期。
    """

    # 这两行代码首先声明了两个全局变量，然后调用一个异步函数 collect() 来获取任务。
    # 声明函数将要修改的全局变量。这些变量可能用于统计已完成和失败的任务数。
    global DONE_TASKS, FAILED_TASKS

    # 调用 collect() 异步函数，并使用 await 关键字等待其返回。根据命名，collect() 很可能负责从 Redis 队列中读取一个任务。它返回两个值：redis_msg（Redis 消息对象）和 task（任务的详细内容，通常是字典形式）。
    # 这是任务处理流程的第一步，collect() 函数封装了从队列中获取任务的复杂逻辑，保持了 handle_task 函数的整洁。
    # 全局变量的使用是为了在整个任务执行器中跟踪任务的成功和失败状态。
    redis_msg, task = await collect()

    # 这是一个检查任务是否为空的条件分支。如果 collect() 没有获取到任务（例如，队列为空），task 变量将为 None。暂停当前任务执行器 5 秒钟，让出 CPU 给其他任务，并避免在没有任务时频繁空转。直接返回，结束本次 handle_task 调用。
    if not task:
        await trio.sleep(5)
        return
    try:
        # 打印任务开始的日志。
        logging.info(f"handle_task begin for task {json.dumps(task)}")

        # 将当前任务的深拷贝（deep copy）存储在一个名为 CURRENT_TASKS 的字典中，通过任务 ID 进行索引。这可能用于追踪正在进行的任务状态。
        CURRENT_TASKS[task["id"]] = copy.deepcopy(task)

        # 调用一个名为 do_handle_task 的异步函数，并等待其完成。这个函数很可能包含了所有具体的任务处理逻辑。
        await do_handle_task(task)

        # 如果 do_handle_task 成功完成，增加已完成任务的计数。计数器用于提供服务状态报告。
        DONE_TASKS += 1

        # 从正在进行的任务字典中移除该任务。
        CURRENT_TASKS.pop(task["id"], None)

        # 打印任务完成的日志。
        logging.info(f"handle_task done for task {json.dumps(task)}")

    # 用于捕获任务处理过程中可能抛出的任何异常。
    # 确保即使任务处理失败，任务执行器也能继续运行，不会崩溃。将失败状态和错误信息写入数据库，使得前端或其他监控工具可以获知任务失败的原因，便于用户和开发者排查。确保无论成功或失败，任务都会从 CURRENT_TASKS 字典中移除。
    except Exception as e:

        # 增加失败任务的计数。
        FAILED_TASKS += 1

        # 即使任务失败，也要将其从正在进行的任务列表中移除。
        CURRENT_TASKS.pop(task["id"], None)
        try:
            err_msg = str(e)

            # 特别处理了 trio 框架中的 ExceptionGroup 异常类型，递归地提取其中的异常信息，将其连接成一个字符串。
            while isinstance(e, exceptiongroup.ExceptionGroup):
                e = e.exceptions[0]
                err_msg += ' -- ' + str(e)

            # 调用 set_progress 函数，将任务的状态更新为失败（-1），并记录详细的错误信息。
            set_progress(task["id"], prog=-1, msg=f"[Exception]: {err_msg}")
        except Exception:
            pass

        # 打印出详细的异常日志，包括完整的栈回溯（traceback）。
        logging.exception(f"handle_task got exception for task {json.dumps(task)}")

    # 确认 Redis 消息。调用从 collect() 函数获取的 Redis 消息对象的 ack() 方法。这会向 Redis 确认该消息已被成功处理。
    # ack() 机制确保了 RAGFlow 即使在处理过程中崩溃，没有 ack 的任务也能被 Redis 重新分配给其他任务执行器，防止任务丢失。这是构建可靠的分布式任务队列的关键。
    redis_msg.ack()


async def report_status():
    global CONSUMER_NAME, BOOT_AT, PENDING_TASKS, LAG_TASKS, DONE_TASKS, FAILED_TASKS
    REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)
    redis_lock = RedisDistributedLock("clean_task_executor", lock_value=CONSUMER_NAME, timeout=60)
    while True:
        try:
            now = datetime.now()
            group_info = REDIS_CONN.queue_info(get_svr_queue_name(0), SVR_CONSUMER_GROUP_NAME)
            if group_info is not None:
                PENDING_TASKS = int(group_info.get("pending", 0))
                LAG_TASKS = int(group_info.get("lag", 0))

            current = copy.deepcopy(CURRENT_TASKS)
            heartbeat = json.dumps({
                "name": CONSUMER_NAME,
                "now": now.astimezone().isoformat(timespec="milliseconds"),
                "boot_at": BOOT_AT,
                "pending": PENDING_TASKS,
                "lag": LAG_TASKS,
                "done": DONE_TASKS,
                "failed": FAILED_TASKS,
                "current": current,
            })
            REDIS_CONN.zadd(CONSUMER_NAME, heartbeat, now.timestamp())
            logging.info(f"{CONSUMER_NAME} reported heartbeat: {heartbeat}")

            expired = REDIS_CONN.zcount(CONSUMER_NAME, 0, now.timestamp() - 60 * 30)
            if expired > 0:
                REDIS_CONN.zpopmin(CONSUMER_NAME, expired)

            # clean task executor
            if redis_lock.acquire():
                task_executors = REDIS_CONN.smembers("TASKEXE")
                for consumer_name in task_executors:
                    if consumer_name == CONSUMER_NAME:
                        continue
                    expired = REDIS_CONN.zcount(
                        consumer_name, now.timestamp() - WORKER_HEARTBEAT_TIMEOUT, now.timestamp() + 10
                    )
                    if expired == 0:
                        logging.info(f"{consumer_name} expired, removed")
                        REDIS_CONN.srem("TASKEXE", consumer_name)
                        REDIS_CONN.delete(consumer_name)
        except Exception:
            logging.exception("report_status got exception")
        finally:
            redis_lock.release()
        await trio.sleep(30)


async def task_manager():
    """
    这是一个异步函数，被 nursery.start_soon 调用，在后台运行。主要作用是提供一个健壮的框架，用于调用 handle_task，并确保在任务完成后（或失败后）正确地释放并发控制的令牌。
    """
    try:
        # 调用 handle_task 异步函数，并等待其完成。
        await handle_task()
    finally:
        # 释放一个信号量或令牌，允许 main 函数中的 task_limiter.acquire() 继续执行，从而启动下一个 task_manager 任务。
        task_limiter.release()


async def main():
    """
    main 函数是一个异步函数的定义，其中包含异步 I/O 操作（如网络请求、文件读写）。trio.run() 会一直运行，直到 main 函数及其所有子任务都完成为止。
    处理并发任务：RAGFlow 的任务执行器需要处理来自队列的多个任务，这通常涉及到大量的 I/O 操作（从 Redis 队列读取、调用模型 API、写入数据库）。使用 trio 这样的异步框架可以高效地处理这些并发 I/O 任务，而无需使用多线程，从而减少上下文切换的开销，提高性能。
    结构化并发：trio 提供了结构化并发的编程模型，使得编写和维护复杂的并发代码更加容易和安全。
    """

    # 打印一个 ASCII 艺术的 RAGFlow logo 到日志。
    logging.info(r"""
  ______           __      ______                     __
 /_  __/___ ______/ /__   / ____/  _____  _______  __/ /_____  _____
  / / / __ `/ ___/ //_/  / __/ | |/_/ _ \/ ___/ / / / __/ __ \/ ___/
 / / / /_/ (__  ) ,<    / /____>  </  __/ /__/ /_/ / /_/ /_/ / /
/_/  \__,_/____/_/|_|  /_____/_/|_|\___/\___/\__,_/\__/\____/_/
    """)

    # 打印版本信息，并初始化和打印配置设置。在服务启动时确认当前运行的版本和配置，是排查问题的第一步。确保配置正确，可以避免许多因配置错误导致的问题。
    # 调用函数获取版本号。
    logging.info(f'TaskExecutor: RAGFlow version: {get_ragflow_version()}')
    # 调用 settings 模块的初始化函数，通常用于从环境变量或配置文件加载和验证配置。
    settings.init_settings()
    # 打印出当前的 RAG 相关配置，以便于调试和确认。
    print_rag_settings()

    # 这段代码块用于配置和启动内存追踪工具 tracemalloc。tracemalloc 是 Python 标准库中一个强大的内存追踪工具，它能帮助开发者定位内存泄漏问题。通过信号动态控制，可以在不重启服务的情况下对内存使用情况进行诊断，这对于长时间运行的服务至关重要。
    # 判断操作系统是否为 Windows，因为 SIGUSR1 和 SIGUSR2 是 Unix-like 系统特有的信号。
    # SIGUSR1 信号会触发 start_tracemalloc_and_snapshot 函数，SIGUSR2 信号会触发 stop_tracemalloc。
    if sys.platform != "win32":
        signal.signal(signal.SIGUSR1, start_tracemalloc_and_snapshot)
        signal.signal(signal.SIGUSR2, stop_tracemalloc)
    # 从环境变量中获取 TRACE_MALLOC_ENABLED 的值。
    TRACE_MALLOC_ENABLED = int(os.environ.get('TRACE_MALLOC_ENABLED', "0"))
    # 如果该环境变量设置为 1，则在服务启动时立即开启内存追踪。
    if TRACE_MALLOC_ENABLED:
        start_tracemalloc_and_snapshot(None, None)

    # 这是为终止信号 SIGINT 和 SIGTERM 注册处理函数。
    # SIGINT 信号通常由 Ctrl+C 触发，SIGTERM 信号是 kill 命令发送的默认终止信号。
    # 这两行代码将这些信号与一个名为 signal_handler 的函数关联起来。当服务需要停止时，它能够执行一个预定义的 signal_handler 函数来清理资源、关闭连接，并优雅地退出，而不是立即被强制终止。
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ==============================================================
    # 这是整个任务执行器的核心异步循环，负责启动状态报告和任务管理。
    # trio 框架中的一个核心概念，它创建了一个“育儿园”（nursery）。在这个 nursery 中启动的所有子任务（start_soon）都会被自动管理。当 nursery 退出时，它会确保所有子任务都已完成或被取消。
    # 结构化并发：trio 的 nursery 模型使得并发编程更加安全和直观。它可以自动处理子任务的生命周期和错误传播。
    # 资源管理和任务调度：report_status 确保了服务的可监控性。task_limiter 确保了任务执行器不会因为同时处理太多任务而过载。
    # 解耦与模块化：将状态报告、任务调度和任务处理逻辑分别封装在不同的异步函数中，使代码结构清晰，易于维护。
    # ==============================================================
    async with trio.open_nursery() as nursery:

        # 在后台启动一个名为 report_status 的异步任务，该任务可能定期向某个地方（如 Redis 或数据库）报告当前任务执行器的健康状态。
        nursery.start_soon(report_status)

        # 这是一个无限循环，只要 stop_event（一个异步事件对象）没有被设置，循环就会一直运行。当 signal_handler 被调用时，通常会设置这个 stop_event，从而中断循环。
        while not stop_event.is_set():

            # 这里使用了task_limiter，这很可能是一个并发控制机制。await 关键字会暂停循环，直到 task_limiter 允许启动新任务（例如，限制同时运行的任务数量）。
            await task_limiter.acquire()

            # 当task_limiter 允许时，在后台启动一个名为 task_manager 的异步任务。task_manager 才是真正负责从队列中拉取和处理任务的核心函数。
            nursery.start_soon(task_manager)

    # 一条错误日志，表示这是一个不应该被执行到的代码行。
    logging.error("BUG!!! You should not reach here!!!")

if __name__ == "__main__":
    # 这是一个 Python 标准库 faulthandler 中的函数调用。faulthandler 模块可以帮助诊断程序崩溃（例如，分段错误 segmentation fault）。
    # faulthandler.enable() 会注册一个处理程序，当 Python 解释器检测到致命错误时，它会打印一个 Python 回溯（traceback），包括所有正在运行的线程的栈信息。
    faulthandler.enable()

    # 这是一个日志初始化函数的调用，可能来自 api.utils.log_utils 模块。它的作用是配置程序的日志系统。
    init_root_logger(CONSUMER_NAME)

    # 这是一个异步框架 trio 的入口点。它启动并运行整个异步应用程序。是整个任务执行器程序的起点。
    # trio.run() 接收一个异步函数（main）作为参数。它会创建一个新的事件循环，并在该循环中执行 main 函数。
    trio.run(main)
