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
from graphrag.utils import (
    get_llm_cache,
    set_llm_cache,
    get_tags_from_cache,
    set_tags_to_cache,
)
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
from rag.app import (
    laws,
    paper,
    presentation,
    manual,
    qa,
    table,
    book,
    resume,
    picture,
    naive,
    one,
    audio,
    email,
    tag,
)
from rag.nlp import search, rag_tokenizer
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.settings import (
    DOC_MAXIMUM_SIZE,
    DOC_BULK_SIZE,
    EMBEDDING_BATCH_SIZE,
    SVR_CONSUMER_GROUP_NAME,
    get_svr_queue_name,
    get_svr_queue_names,
    print_rag_settings,
    TAG_FLD,
    PAGERANK_FLD,
)
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
    ParserType.TAG.value: tag,
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

MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "5"))
MAX_CONCURRENT_CHUNK_BUILDERS = int(
    os.environ.get("MAX_CONCURRENT_CHUNK_BUILDERS", "1")
)
MAX_CONCURRENT_MINIO = int(os.environ.get("MAX_CONCURRENT_MINIO", "10"))
task_limiter = trio.Semaphore(MAX_CONCURRENT_TASKS)
chunk_limiter = trio.CapacityLimiter(MAX_CONCURRENT_CHUNK_BUILDERS)
embed_limiter = trio.CapacityLimiter(MAX_CONCURRENT_CHUNK_BUILDERS)
minio_limiter = trio.CapacityLimiter(MAX_CONCURRENT_MINIO)
kg_limiter = trio.CapacityLimiter(2)
WORKER_HEARTBEAT_TIMEOUT = int(os.environ.get("WORKER_HEARTBEAT_TIMEOUT", "120"))
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
    snapshot_file = os.path.abspath(
        os.path.join(
            get_project_base_directory(),
            "logs",
            f"{os.getpid()}_snapshot_{timestamp}.trace",
        )
    )

    snapshot = tracemalloc.take_snapshot()
    snapshot.dump(snapshot_file)
    current, peak = tracemalloc.get_traced_memory()
    if sys.platform == "win32":
        import psutil

        process = psutil.Process()
        max_rss = process.memory_info().rss / 1024
    else:
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info(
        f"taken snapshot {snapshot_file}. max RSS={max_rss / 1000:.2f} MB, current memory usage: {current / 10**6:.2f} MB, Peak memory usage: {peak / 10**6:.2f} MB"
    )


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
        logging.exception(
            f"set_progress({task_id}), progress: {prog}, progress_msg: {msg}, got exception"
        )


async def collect():
    global CONSUMER_NAME, DONE_TASKS, FAILED_TASKS
    global UNACKED_ITERATOR

    svr_queue_names = get_svr_queue_names()
    try:
        if not UNACKED_ITERATOR:
            UNACKED_ITERATOR = REDIS_CONN.get_unacked_iterator(
                svr_queue_names, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME
            )
        try:
            redis_msg = next(UNACKED_ITERATOR)
        except StopIteration:
            for svr_queue_name in svr_queue_names:
                redis_msg = REDIS_CONN.queue_consumer(
                    svr_queue_name, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME
                )
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


# @timeout(60*40, 1) 是一个装饰器，为函数设置了 40 分钟的超时时间。
@timeout(60 * 40, 1)
async def build_chunks(task, progress_callback):
    ## 第1步：文件大小检查
    #  是一个前置检查，确保文件大小不超过配置的最大值。
    if task["size"] > DOC_MAXIMUM_SIZE:
        set_progress(
            task["id"],
            prog=-1,
            msg="File size exceeds( <= %dMb )" % (int(DOC_MAXIMUM_SIZE / 1024 / 1024)),
        )
        return []

    ## 第2步：获取解析器与文件二进制内容
    # FACTORY 是一个字典或映射，用于根据解析器 ID 查找对应的解析器类。
    # 负责从数据库中获取文件存储位置，并从 MinIO（对象存储）下载文件内容。
    # 使用 task 字典中的 parser_id 作为键，从 FACTORY 中获取对应的解析器对象。这是一种策略模式的实现。
    # 通过 FACTORY 机制，可以轻松地添加新的文件解析器（如新的文档格式），而无需修改 build_chunks 函数本身。
    # 将文件存储地址的获取和文件的下载逻辑分开，代码结构更清晰。使用异步下载可以避免阻塞事件循环。
    chunker = FACTORY[task["parser_id"].lower()]
    try:
        st = timer()

        # 调用数据库服务获取文件在 MinIO 中的桶（bucket）和对象名（name）。
        bucket, name = File2DocumentService.get_storage_address(doc_id=task["doc_id"])

        # 一个异步函数，负责从 MinIO 异步下载文件内容到内存，返回二进制数据。
        binary = await get_storage_binary(bucket, name)
        logging.info(
            "From minio({}) {}/{}".format(timer() - st, task["location"], task["name"])
        )
    except TimeoutError:
        progress_callback(
            -1,
            "Internal server error: Fetch file from minio timeout. Could you try it again.",
        )
        logging.exception(
            "Minio {}/{} got timeout: Fetch file from minio timeout.".format(
                task["location"], task["name"]
            )
        )
        raise
    except Exception as e:
        if re.search("(No such file|not found)", str(e)):
            progress_callback(
                -1,
                "Can not find file <%s> from minio. Could you try it again?"
                % task["name"],
            )
        else:
            progress_callback(-1, "Get file from minio: %s" % str(e).replace("'", ""))
        logging.exception(
            "Chunking {}/{} got exception".format(task["location"], task["name"])
        )
        raise

    ## 第3步： 调用具体解析器进行分块的核心部分。
    try:
        # 使用并发限制器，控制同时进行分块的线程数量，防止内存过载。
        async with chunk_limiter:

            # 因为大多数解析器库（例如处理 PDF 的 PyMuPDF、处理 PPT 的 python-pptx 等）都是同步阻塞的，所以 RAGFlow 使用 trio 框架将同步的 chunker.chunk 方法放到一个单独的线程中运行。
            # 调用在第二步获取的 chunker 对象的 chunk 方法，传入文件名、二进制内容、页码范围等参数，返回一系列文本块。
            cks = await trio.to_thread.run_sync(
                lambda: chunker.chunk(
                    task["name"],
                    binary=binary,
                    from_page=task["from_page"],
                    to_page=task["to_page"],
                    lang=task["language"],
                    callback=progress_callback,
                    kb_id=task["kb_id"],
                    parser_config=task["parser_config"],
                    tenant_id=task["tenant_id"],
                )
            )
        logging.info(
            "Chunking({}) {}/{} done".format(
                timer() - st, task["location"], task["name"]
            )
        )
    except TaskCanceledException:
        raise
    except Exception as e:
        progress_callback(
            -1, "Internal server error while chunking: %s" % str(e).replace("'", "")
        )
        logging.exception(
            "Chunking {}/{} got exception".format(task["location"], task["name"])
        )
        raise

    ## 第4步：这部分代码负责处理 chunker 返回的原始文本块 cks，为每个文本块添加必要的元数据，并异步上传可能包含的图片。
    # 初始化一个空列表，用于存储处理完成后的文档块。
    docs = []
    doc = {"doc_id": task["doc_id"], "kb_id": str(task["kb_id"])}
    if task["pagerank"]:
        doc[PAGERANK_FLD] = int(task["pagerank"])
    st = timer()

    # @timeout(60)是装饰器，这表示每次执行这个函数，如果 60 秒内没完成，就会超时抛错。
    @timeout(60)
    async def upload_to_minio(document, chunk):
        """
        这是一个内部定义的异步函数upload_to_minio，它会为每个文本块生成唯一的 ID，处理其中的图片（转换为 JPEG 格式），并异步上传到 MinIO，然后记录到 docs 供后续关键词、问题生成、打标签等步骤使用。
        输入参数：
        - document: 文档块对象，包含文本块的元数据。一个字典，包含文档的元信息，比如 doc_id, kb_id, pagerank 等。
        - chunk: 文本块内容。一个字典，表示文档的某个分块。
        输出参数：无（返回值 None）
        """
        try:
            d = copy.deepcopy(document)
            d.update(chunk)
            d["id"] = xxhash.xxh64(
                (chunk["content_with_weight"] + str(d["doc_id"])).encode("utf-8")
            ).hexdigest()
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
                    d["image"].save(output_buffer, format="JPEG")

                async with minio_limiter:
                    # 这里调用 STORAGE_IMPL.put(...) 把图片二进制数据上传到 MinIO。这是另一个副作用（网络/存储操作），不依赖返回值传给调用者。
                    await trio.to_thread.run_sync(
                        lambda: STORAGE_IMPL.put(
                            task["kb_id"], d["id"], output_buffer.getvalue()
                        )
                    )
                d["img_id"] = "{}-{}".format(task["kb_id"], d["id"])
                if not isinstance(d["image"], bytes):
                    d["image"].close()
                del d["image"]  # Remove image reference
                docs.append(d)
            finally:
                output_buffer.close()  # Ensure BytesIO is always closed
        except Exception:
            logging.exception(
                "Saving image of chunk {}/{}/{} got exception".format(
                    task["location"], task["name"], d["id"]
                )
            )
            raise

    async with trio.open_nursery() as nursery:
        for ck in cks:
            nursery.start_soon(upload_to_minio, doc, ck)

    el = timer() - st
    logging.info("MINIO PUT({}) cost {:.3f} s".format(task["name"], el))

    ## 第5步：高级处理功能：关键字、问题和标签生成。这些操作都通过 trio.open_nursery() 实现了并发处理，
    ## 并且使用了 chat_limiter 来控制对 LLM 的调用频率，以及使用了缓存（get_llm_cache）来避免重复调用。
    ## 为什么？
    ## 增强检索能力：这些元数据（关键字、问题、标签）可以作为强大的附加信息，提高检索阶段的准确性和多样性，帮助 RAG 系统更好地理解和匹配用户查询。
    ## 提高效率：使用缓存和并发调用 LLM，可以在保证功能的同时，尽可能地减少处理时间。
    # 如果配置了 auto_keywords，代码会调用 keyword_extraction 函数，让 LLM 从每个文本块中提取出核心关键字。
    if task["parser_config"].get("auto_keywords", 0):
        st = timer()
        progress_callback(msg="Start to generate keywords for every chunk ...")
        chat_mdl = LLMBundle(
            task["tenant_id"],
            LLMType.CHAT,
            llm_name=task["llm_id"],
            lang=task["language"],
        )

        async def doc_keyword_extraction(chat_mdl, d, topn):
            cached = get_llm_cache(
                chat_mdl.llm_name, d["content_with_weight"], "keywords", {"topn": topn}
            )
            if not cached:
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(
                        lambda: keyword_extraction(
                            chat_mdl, d["content_with_weight"], topn
                        )
                    )
                set_llm_cache(
                    chat_mdl.llm_name,
                    d["content_with_weight"],
                    cached,
                    "keywords",
                    {"topn": topn},
                )
            if cached:
                d["important_kwd"] = cached.split(",")
                d["important_tks"] = rag_tokenizer.tokenize(
                    " ".join(d["important_kwd"])
                )
            return

        async with trio.open_nursery() as nursery:
            for d in docs:
                nursery.start_soon(
                    doc_keyword_extraction,
                    chat_mdl,
                    d,
                    task["parser_config"]["auto_keywords"],
                )
        progress_callback(
            msg="Keywords generation {} chunks completed in {:.2f}s".format(
                len(docs), timer() - st
            )
        )

    # 如果配置了 auto_questions，代码会调用 question_proposal 函数，让 LLM 根据文本块内容生成可能的问题。
    if task["parser_config"].get("auto_questions", 0):
        st = timer()
        progress_callback(msg="Start to generate questions for every chunk ...")
        chat_mdl = LLMBundle(
            task["tenant_id"],
            LLMType.CHAT,
            llm_name=task["llm_id"],
            lang=task["language"],
        )

        async def doc_question_proposal(chat_mdl, d, topn):
            cached = get_llm_cache(
                chat_mdl.llm_name, d["content_with_weight"], "question", {"topn": topn}
            )
            if not cached:
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(
                        lambda: question_proposal(
                            chat_mdl, d["content_with_weight"], topn
                        )
                    )
                set_llm_cache(
                    chat_mdl.llm_name,
                    d["content_with_weight"],
                    cached,
                    "question",
                    {"topn": topn},
                )
            if cached:
                d["question_kwd"] = cached.split("\n")
                d["question_tks"] = rag_tokenizer.tokenize("\n".join(d["question_kwd"]))

        async with trio.open_nursery() as nursery:
            for d in docs:
                nursery.start_soon(
                    doc_question_proposal,
                    chat_mdl,
                    d,
                    task["parser_config"]["auto_questions"],
                )
        progress_callback(
            msg="Question generation {} chunks completed in {:.2f}s".format(
                len(docs), timer() - st
            )
        )

    # 如果配置了 tag_kb_ids，代码会调用 content_tagging 函数，让 LLM 根据预设的标签体系对文本块进行分类和打标签。
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

        chat_mdl = LLMBundle(
            task["tenant_id"],
            LLMType.CHAT,
            llm_name=task["llm_id"],
            lang=task["language"],
        )

        docs_to_tag = []
        for d in docs:
            task_canceled = has_canceled(task["id"])
            if task_canceled:
                progress_callback(-1, msg="Task has been canceled.")
                return
            if (
                settings.retrievaler.tag_content(
                    tenant_id, kb_ids, d, all_tags, topn_tags=topn_tags, S=S
                )
                and len(d[TAG_FLD]) > 0
            ):
                examples.append(
                    {"content": d["content_with_weight"], TAG_FLD: d[TAG_FLD]}
                )
            else:
                docs_to_tag.append(d)

        async def doc_content_tagging(chat_mdl, d, topn_tags):
            cached = get_llm_cache(
                chat_mdl.llm_name,
                d["content_with_weight"],
                all_tags,
                {"topn": topn_tags},
            )
            if not cached:
                picked_examples = (
                    random.choices(examples, k=2) if len(examples) > 2 else examples
                )
                if not picked_examples:
                    picked_examples.append(
                        {"content": "This is an example", TAG_FLD: {"example": 1}}
                    )
                async with chat_limiter:
                    cached = await trio.to_thread.run_sync(
                        lambda: content_tagging(
                            chat_mdl,
                            d["content_with_weight"],
                            all_tags,
                            picked_examples,
                            topn=topn_tags,
                        )
                    )
                if cached:
                    cached = json.dumps(cached)
            if cached:
                set_llm_cache(
                    chat_mdl.llm_name,
                    d["content_with_weight"],
                    cached,
                    all_tags,
                    {"topn": topn_tags},
                )
                d[TAG_FLD] = json.loads(cached)

        async with trio.open_nursery() as nursery:
            for d in docs_to_tag:
                nursery.start_soon(doc_content_tagging, chat_mdl, d, topn_tags)
        progress_callback(
            msg="Tagging {} chunks completed in {:.2f}s".format(len(docs), timer() - st)
        )

    ## 第6步：函数返回一个包含所有处理完成、带有元数据和潜在图片的文档块列表。
    return docs


def init_kb(row, vector_size: int):
    idxnm = search.index_name(row["tenant_id"])
    return settings.docStoreConn.createIdx(idxnm, row.get("kb_id", ""), vector_size)


## 第1步：一个装饰器，为函数设置了 20 分钟的超时时间。
@timeout(60 * 20)
async def embedding(docs, mdl, parser_config=None, callback=None):
    """
    embedding异步函数，接收文档块列表 docs、嵌入模型 mdl、解析器配置 parser_config 和回调函数 callback。
    """
    if parser_config is None:
        parser_config = {}

    ## 第2步：准备嵌入文本数据
    """
    这部分代码遍历所有文档块 docs，准备两组文本数据用于嵌入：
    tts：用于存储文档的标题或关键字。
    cnts：用于存储文档块的主要内容。
    """
    tts, cnts = [], []
    for d in docs:
        # 从文档块中获取文档名称或关键字，如果没有则使用默认值 "Title"。
        tts.append(d.get("docnm_kwd", "Title"))

        # 这是关键。它优先使用 build_chunks 生成的“问题”列表作为嵌入文本。这是 RAGFlow 的一个重要优化策略，如果文档块有生成问题，就用问题来代表这个文档块的内容。
        # 为什么？使用“问题”作为嵌入文本，可以让生成的向量更贴近用户可能提出的问题，从而在检索时获得更精准的匹配。这是一种“查询-文档”匹配的优化策略。
        c = "\n".join(d.get("question_kwd", []))

        # 如果文档块没有生成问题，则回退到使用文档块的原始内容。
        if not c:
            c = d["content_with_weight"]

        # 使用正则表达式移除 HTML 表格标签，对文本进行预处理。
        # 为什么？移除 HTML 标签可以避免无用信息干扰嵌入模型的编码效果。
        c = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", c)
        if not c:
            c = "None"
        cnts.append(c)

    ## 第3步：文档标题（TTS）的嵌入与复制
    # 这部分代码用于处理文档标题的向量。它只对第一个标题进行一次编码，然后复制该向量以匹配所有文档块。
    tk_count = 0
    if len(tts) == len(cnts):
        # 只编码 tts 列表中的第一个元素。
        # 为什么？假设同一个文档的所有分块都共享相同的文档标题，那么只需编码一次标题即可，避免了重复的、耗时的模型调用，显著提高了效率。
        vts, c = await trio.to_thread.run_sync(lambda: mdl.encode(tts[0:1]))

        # 将编码后的单个向量复制 len(tts) 次，生成一个与 cnts 长度相同的向量数组。
        # 通过复制，确保标题向量数组的维度与内容向量数组的维度一致，为后续的向量融合做准备。
        tts = np.concatenate([vts for _ in range(len(tts))], axis=0)
        tk_count += c

    ## 第4步：内容（CNTS）的批量嵌入。这是核心的批量嵌入逻辑。
    cnts_ = np.array([])
    # 一个循环，按照 EMBEDDING_BATCH_SIZE 的大小，对 cnts 列表进行分批处理。
    for i in range(0, len(cnts), EMBEDDING_BATCH_SIZE):

        # 使用并发限制器，控制同时进行嵌入的批量任务数量，防止模型服务过载。
        # 为什么？embed_limiter 和 下面的truncate 确保了任务执行器能够稳定高效地运行，不会因为一次性处理过多数据而崩溃。
        async with embed_limiter:
            # 同样，将同步的 mdl.encode 方法放入线程中执行。
            vts, c = await trio.to_thread.run_sync(
                # 调用嵌入模型的 encode 方法，传入一个批量的文本列表。
                lambda: mdl.encode(
                    [
                        # 在编码前，对文本内容进行截断，以确保文本长度不超过模型的最大输入长度。
                        truncate(c, mdl.max_length - 10)
                        for c in cnts[i : i + EMBEDDING_BATCH_SIZE]
                    ]
                )
            )
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            # 将每个批次返回的向量数组拼接成一个大的数组。
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    ## 第5步：向量加权融合。这部分代码实现了文档标题向量和内容向量的加权融合。
    # 从任务配置中获取权重，如果未设置，则默认为 0.1。
    filename_embd_weight = parser_config.get(
        "filename_embd_weight", 0.1
    )  # due to the db support none value
    if not filename_embd_weight:
        filename_embd_weight = 0.1
    title_w = float(filename_embd_weight)

    # 使用一个简单的数学公式进行加权求和，将标题向量 tts 和内容向量 cnts 融合，生成最终的向量 vects。
    # title_w * tts 代表标题的权重，(1 - title_w) * cnts 代表内容的权重。
    # 为什么？在检索时，文档标题（或关键字）通常包含非常重要的信息。通过将标题向量与内容向量进行加权融合，可以使最终的向量同时代表文档的核心主题和具体内容，从而提高检索的准确性。
    # 这是一种非常常见的 RAG 优化技术。
    vects = (title_w * tts + (1 - title_w) * cnts) if len(tts) == len(cnts) else cnts

    ## 第6步： 数据封装与返回。这是 embedding 函数的最后一步，将生成的向量封装回文档块中，并返回总的 token 计数和向量维度。
    # 断言确保向量数量与文档块数量一致，这是一个重要的完整性检查。
    assert len(vects) == len(docs)
    vector_size = 0
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        vector_size = len(v)
        # 将生成的向量 v 附加到每个文档块 d 中。键名 q_..._vec 是一个动态生成的名称，包含了向量的维度，以便后续数据库插入和检索。
        d["q_%d_vec" % len(v)] = v
    # 返回总的 token 计数和向量维度，这些信息将在 do_handle_task 函数中用于更新数据库元数据。
    # 为什么？状态持久化。返回的 tk_count 和 vector_size 都是重要的元数据，需要被记录到数据库中。
    return tk_count, vector_size


@timeout(3600)
async def run_raptor(row, chat_mdl, embd_mdl, vector_size, callback=None):
    # Pressure test for GraphRAG task
    await is_strong_enough(chat_mdl, embd_mdl)
    chunks = []
    vctr_nm = "q_%d_vec" % vector_size
    for d in settings.retrievaler.chunk_list(
        row["doc_id"],
        row["tenant_id"],
        [str(row["kb_id"])],
        fields=["content_with_weight", vctr_nm],
    ):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"],
    )
    original_length = len(chunks)
    chunks = await raptor(
        chunks, row["parser_config"]["raptor"]["random_seed"], callback
    )
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"]),
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


# @timeout(60*60, 1) 是一个装饰器，它为 do_handle_task 函数设置了超时机制。这个装饰器来自 api.utils.api_utils，它会监控函数的执行时间。如果函数执行超过 3600 秒（1 小时），它会抛出 TimeoutError。
# 为什么？文档处理是一个耗时操作，可能会因为文件损坏、外部服务无响应等原因而卡住。设置超时可以确保任务不会永久阻塞，从而保证任务执行器的可用性。
@timeout(60 * 60, 1)
async def do_handle_task(task):
    """
    负责编排整个文档处理和索引流程。这段代码涵盖了从任务准备、模型绑定、取消检查、到实际的文档分块、向量嵌入和数据存储等多个关键步骤。
    类比：把 do_handle_task 想象成一个工厂的生产线经理，它的任务是监督一个文档从“原材料”到“成品”的全过程。
    这个生产线经理的核心思路可以分为以下几个关键步骤：
    步骤 1：准备工作与前置检查
        在正式开始生产之前，经理需要先做一些准备和检查，以确保生产过程能够顺利进行。
        提取任务指令：从 task 字典中拿出所有关键信息，比如任务 ID、文档名称、LLM 模型 ID 等。这相当于拿到一份详细的生产工单。
        配置进度报告：设置一个 progress_callback 函数，这个函数就像生产线上的实时看板，随时可以报告当前的生产进度。
        检查是否已取消：在开始耗时工作前，先检查一下工单是否被上级取消了。如果取消了，就立即停止，避免浪费资源。
        绑定和测试模型：工厂生产需要用到一些关键工具，比如嵌入模型。在开始前，经理会先测试一下这些工具是否正常工作，并确认它们的规格（比如向量维度）。如果工具坏了，就立即报错。
    步骤 2：任务类型分发与核心生产流程
        这是整个生产线的核心。经理需要根据工单上的“产品类型”，选择不同的生产流程。
        产品类型识别：通过检查 task_type（例如 raptor、graphrag 或空值），确定文档要走哪条生产线。
        选择生产线：
        如果产品是 raptor 或 graphrag：这是两条特殊的高级生产线。经理会调用 run_raptor 或 run_graphrag 函数，启动对应的复杂流程。这些流程有自己的内部逻辑，可能需要调用额外的工具（比如 LLM）来完成。
        如果产品是标准类型：这是最常见的生产线。经理会按部就班地执行以下几个子步骤：
        切割原材料：调用 build_chunks 函数，将原始文档（原材料）切割成一个个小的文本块（半成品）。
        生成核心组件：调用 embedding 函数，为每一个文本块生成一个向量（核心组件）。这个过程非常耗时，但至关重要。
        实时报告进度：在完成每一步后，经理都会通过 progress_callback 更新生产线上的进度看板。
    步骤 3：成品入库与收尾工作
        生产完成后，经理需要将成品妥善地存储，并做好收尾工作。
        批量入库：将所有切割好的文本块及其生成的向量，分批次地插入到文档存储中（比如 Elasticsearch）。这里使用批量操作是为了提高效率，就像一次性搬运很多箱子，而不是一箱一箱地搬。
        处理异常：在入库过程中，如果遇到问题（比如数据库连接失败），经理会立即停止，并记录错误信息。
        更新库存记录：将入库成功的文本块 ID 记录下来，并更新总的库存数量、token 数量等元数据。这相当于更新了总账本。
        最终报告：生产线经理最后会通过进度看板，宣布本次生产任务已完成，并报告总耗时，整个流程正式结束。
    总结一下：
        do_handle_task 函数的思路，就是以一个总控制器的角色，负责编排、监督和管理整个文档处理的生命周期。它不负责具体的“切割”或“生成向量”工作（这些由子函数 build_chunks 和 embedding 完成），而是确保每个步骤都按顺序、按规则执行，同时处理好可能出现的异常和中断。
    """

    ## 第1步： task_id 等变量的赋值是从传入的 task 字典中提取任务信息。
    # 将字典中的值提取到局部变量中，使得后续代码在引用这些任务参数时更加清晰和简洁。
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
    """
    partial来自 functools 模块，它的作用是固定（预先填充）一个函数的部分参数，返回一个新的可调用对象。它创建一个新的函数 progress_callback，其中一些参数（task_id, task_from_page, task_to_page）已经被固定。
    partial 使得在代码中更新进度变得非常简单，只需调用 progress_callback(prog=...) 即可，而无需每次都传参数（task_id, task_from_page, task_to_page）。
    原本 set_progress 可能是这样定义的：
    def set_progress(task_id, task_from_page, task_to_page, prog=None, msg=""): 
        ...
    用 partial 把前三个参数 task_id、task_from_page、task_to_page 固定住了。
    等价于：
    def progress_callback(prog=None, msg=""):
        return set_progress(task_id, task_from_page, task_to_page, prog, msg)"""
    progress_callback = partial(set_progress, task_id, task_from_page, task_to_page)

    # FIXME: workaround, Infinity doesn't support table parsing method, this check is to notify user
    lower_case_doc_engine = settings.DOC_ENGINE.lower()
    if lower_case_doc_engine == "infinity" and task["parser_id"].lower() == "table":
        error_message = "Table parsing method is not supported by Infinity, please use other parsing methods or use Elasticsearch as the document engine."
        progress_callback(-1, msg=error_message)
        raise Exception(error_message)

    # has_canceled 是一个检查任务是否被取消的函数。
    # 调用函数查询数据库或 Redis，检查特定 task_id 的任务是否被标记为取消。
    # RAGFlow 允许用户取消正在进行的任务。在耗时操作前检查任务状态可以及时终止任务，节省计算资源。
    task_canceled = has_canceled(task_id)

    # 如果任务被取消，则返回错误消息。
    if task_canceled:
        progress_callback(-1, msg="Task has been canceled.")
        return

    ## 第2步：这部分代码负责绑定和初始化嵌入模型，并获取其输出向量的维度。
    # 在正式处理任务之前，先检查和绑定模型可以提前发现配置错误或模型服务不可用的问题，避免后续不必要的计算。
    try:
        # bind embedding model
        # 实例化一个 LLMBundle 对象，它封装了对嵌入模型的调用，并根据任务配置（如租户 ID、模型名称等）进行初始化。
        embedding_model = LLMBundle(
            task_tenant_id,
            LLMType.EMBEDDING,
            llm_name=task_embedding_id,
            lang=task_language,
        )

        # 这是一个检查函数，确保模型可用且符合最低要求。
        await is_strong_enough(None, embedding_model)

        # 用一个简单的字符串 "ok" 来调用模型，以确保其正常工作，并获取返回的向量。
        vts, _ = embedding_model.encode(["ok"])

        # 获取向量维度，这对于创建数据库索引至关重要。
        vector_size = len(vts[0])
    except Exception as e:
        error_message = f"Fail to bind embedding model: {str(e)}"
        progress_callback(-1, msg=error_message)
        logging.exception(error_message)
        raise

    init_kb(task, vector_size)

    # Either using RAPTOR or Standard chunking methods
    ## 第3步：这是 do_handle_task 函数中最重要的分发逻辑，它根据任务类型（task_type）调用不同的处理流程。
    ## 模块化和扩展性：这种分发机制将不同的任务处理方法解耦，使得 RAGFlow 能够轻松支持多种文档处理策略，例如 RAPTOR 和 GraphRAG。
    ## 单一职责：每个代码块只负责一种特定类型的任务处理，使得代码更加清晰和易于维护。
    # raptor 任务：调用 run_raptor 函数，这是一种高级的文档处理方法，通常用于构建多层级摘要和索引。
    if task.get("task_type", "") == "raptor":
        # bind LLM for raptor
        chat_model = LLMBundle(
            task_tenant_id, LLMType.CHAT, llm_name=task_llm_id, lang=task_language
        )
        await is_strong_enough(chat_model, None)
        # run RAPTOR
        async with kg_limiter:
            chunks, token_count = await run_raptor(
                task, chat_model, embedding_model, vector_size, progress_callback
            )
    # Either using graphrag or Standard chunking methods
    # graphrag 任务：调用 run_graphrag 函数，用于构建知识图谱。
    elif task.get("task_type", "") == "graphrag":
        if not task_parser_config.get("graphrag", {}).get("use_graphrag", False):
            progress_callback(prog=-1.0, msg="Internal configuration error.")
            return
        graphrag_conf = task["kb_parser_config"].get("graphrag", {})
        start_ts = timer()
        chat_model = LLMBundle(
            task_tenant_id, LLMType.CHAT, llm_name=task_llm_id, lang=task_language
        )
        await is_strong_enough(chat_model, None)
        with_resolution = graphrag_conf.get("resolution", False)
        with_community = graphrag_conf.get("community", False)
        async with kg_limiter:
            await run_graphrag(
                task,
                task_language,
                with_resolution,
                with_community,
                chat_model,
                embedding_model,
                progress_callback,
            )
        progress_callback(
            prog=1.0, msg="Knowledge Graph done ({:.2f}s)".format(timer() - start_ts)
        )
        return
    else:
        # Standard chunking methods
        # 标准任务：调用 build_chunks 和 embedding 等函数，这是标准文档处理流程的核心步骤，包括文档分块和生成嵌入向量。
        start_ts = timer()

        ## 第3.1步：调用 build_chunks 函数，它会根据任务配置（例如，解析器类型 parser_id）从存储中下载文件，并调用相应的解析器进行文档分块。
        chunks = await build_chunks(task, progress_callback)
        logging.info(
            "Build document {}: {:.2f}s".format(task_document_name, timer() - start_ts)
        )
        if not chunks:
            progress_callback(1.0, msg=f"No chunk built from {task_document_name}")
            return
        # TODO: exception handler
        ## set_progress(task["did"], -1, "ERROR: ")
        progress_callback(msg="Generate {} chunks".format(len(chunks)))
        start_ts = timer()
        try:
            ## 第3.2步：调用 embedding 函数，将文本块转化为带有向量的 chunks 列表。
            token_count, vector_size = await embedding(
                chunks, embedding_model, task_parser_config, progress_callback
            )
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
                "Deleting image of chunk {}/{}/{} got exception".format(
                    task["location"], task["name"], chunk_id
                )
            )
            raise

    ## 第4步：这部分代码负责将分块好的文档及其向量批量插入到文档存储中（如 Elasticsearch 或 Infinity）。
    ## do_handle_task 遍历 embedding 返回的 chunks 列表，并调用 settings.docStoreConn.insert() 方法，将带有向量的文档块批量插入到文档存储（向量数据库）中。
    # for b in range(...)：一个循环，将文档分块（chunks）切片，以 DOC_BULK_SIZE 为单位进行批量处理。
    for b in range(0, len(chunks), DOC_BULK_SIZE):

        # 使用了 trio 框架的线程转换功能。docStoreConn.insert 方法可能是一个同步阻塞操作，trio.to_thread.run_sync 将其放在一个单独的线程中运行，以避免阻塞整个异步事件循环。
        doc_store_result = await trio.to_thread.run_sync(
            lambda: settings.docStoreConn.insert(
                chunks[b : b + DOC_BULK_SIZE],
                search.index_name(task_tenant_id),
                task_dataset_id,
            )
        )
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
        chunk_ids = [chunk["id"] for chunk in chunks[: b + DOC_BULK_SIZE]]
        chunk_ids_str = " ".join(chunk_ids)
        try:
            TaskService.update_chunk_ids(task["id"], chunk_ids_str)
        except DoesNotExist:
            logging.warning(
                f"do_handle_task update_chunk_ids failed since task {task['id']} is unknown."
            )
            doc_store_result = await trio.to_thread.run_sync(
                lambda: settings.docStoreConn.delete(
                    {"id": chunk_ids},
                    search.index_name(task_tenant_id),
                    task_dataset_id,
                )
            )
            async with trio.open_nursery() as nursery:
                for chunk_id in chunk_ids:
                    nursery.start_soon(delete_image, task_dataset_id, chunk_id)
            progress_callback(
                -1, msg=f"Chunk updates failed since task {task['id']} is unknown."
            )
            return

    logging.info(
        "Indexing doc({}), page({}-{}), chunks({}), elapsed: {:.2f}".format(
            task_document_name,
            task_from_page,
            task_to_page,
            len(chunks),
            timer() - start_ts,
        )
    )

    ## 第5步：调用数据库服务，更新文档和知识库的元数据，例如分块数量和 token 数量。
    DocumentService.increment_chunk_num(
        task_doc_id, task_dataset_id, token_count, chunk_count, 0
    )

    time_cost = timer() - start_ts
    task_time_cost = timer() - task_start_ts

    ## 第6步：在任务完成时，将进度设置为 1.0，并打印最终的完成信息。
    progress_callback(
        prog=1.0,
        msg="Indexing done ({:.2f}s). Task done ({:.2f}s)".format(
            time_cost, task_time_cost
        ),
    )
    logging.info(
        "Chunk doc({}), page({}-{}), chunks({}), token({}), elapsed:{:.2f}".format(
            task_document_name,
            task_from_page,
            task_to_page,
            len(chunks),
            token_count,
            task_time_cost,
        )
    )


async def handle_task():
    """
    这是一个异步函数，负责获取一个任务并处理它，同时处理成功、失败和无任务的情况。
    handle_task 函数是一个非常重要的控制器。它负责获取任务、处理任务、处理任务的成功和失败状态、并最终向消息队列确认处理完成，从而构成了 RAGFlow 任务执行器中一个完整、健壮、可靠的任务处理生命周期。
    """

    # 这两行代码首先声明了两个全局变量，然后调用一个异步函数 collect() 来获取任务。
    # 声明函数将要修改的全局变量。这些变量可能用于统计已完成和失败的任务数。
    global DONE_TASKS, FAILED_TASKS

    ## 第1步：调用 collect() 从 Redis 队列中获取一个任务。
    # 调用 collect() 异步函数，并使用 await 关键字等待其返回。根据命名，collect() 很可能负责从 Redis 队列中读取一个任务。它返回两个值：redis_msg（Redis 消息对象）和 task（任务的详细内容，通常是字典形式）。
    # 这是任务处理流程的第一步，collect() 函数封装了从队列中获取任务的复杂逻辑，保持了 handle_task 函数的整洁。
    # 全局变量的使用是为了在整个任务执行器中跟踪任务的成功和失败状态。
    redis_msg, task = await collect()

    ## 第2步：如果队列为空，休眠 5 秒后返回。
    # 这是一个检查任务是否为空的条件分支。如果 collect() 没有获取到任务（例如，队列为空），task 变量将为 None。暂停当前任务执行器 5 秒钟，让出 CPU 给其他任务，并避免在没有任务时频繁空转。直接返回，结束本次 handle_task 调用。
    if not task:
        await trio.sleep(5)
        return

    ## 第3步：处理任务。
    try:
        # 打印任务开始的日志。
        logging.info(f"handle_task begin for task {json.dumps(task)}")

        # 将当前任务的深拷贝（deep copy）存储在一个名为 CURRENT_TASKS 的字典中，通过任务 ID 进行索引。这可能用于追踪正在进行的任务状态。
        CURRENT_TASKS[task["id"]] = copy.deepcopy(task)

        # 调用一个名为 do_handle_task 的异步函数，并等待其完成。这个函数包含了所有具体的任务处理逻辑。
        await do_handle_task(task)

        ## 第4步：如果 do_handle_task() 成功，更新成功计数并记录日志。
        # 如果 do_handle_task 成功完成，增加已完成任务的计数。计数器用于提供服务状态报告。
        DONE_TASKS += 1

        # 从正在进行的任务字典中移除该任务。
        CURRENT_TASKS.pop(task["id"], None)

        # 打印任务完成的日志。
        logging.info(f"handle_task done for task {json.dumps(task)}")

    ## 第5步：如果 do_handle_task() 失败，捕获异常，更新失败计数，格式化错误信息并更新数据库中的任务状态。
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
                err_msg += " -- " + str(e)

            # 调用 set_progress 函数，将任务的状态更新为失败（-1），并记录详细的错误信息。
            set_progress(task["id"], prog=-1, msg=f"[Exception]: {err_msg}")
        except Exception:
            pass

        # 打印出详细的异常日志，包括完整的栈回溯（traceback）。
        logging.exception(f"handle_task got exception for task {json.dumps(task)}")

    ## 第6步：确认 Redis 消息。向 Redis 确认（ack）任务已被处理。
    # 确认 Redis 消息。调用从 collect() 函数获取的 Redis 消息对象的 ack() 方法。这会向 Redis 确认该消息已被成功处理。
    # ack() 机制确保了 RAGFlow 即使在处理过程中崩溃，没有 ack 的任务也能被 Redis 重新分配给其他任务执行器，防止任务丢失。这是构建可靠的分布式任务队列的关键。
    redis_msg.ack()


async def report_status():
    global CONSUMER_NAME, BOOT_AT, PENDING_TASKS, LAG_TASKS, DONE_TASKS, FAILED_TASKS
    REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)
    redis_lock = RedisDistributedLock(
        "clean_task_executor", lock_value=CONSUMER_NAME, timeout=60
    )
    while True:
        try:
            now = datetime.now()
            group_info = REDIS_CONN.queue_info(
                get_svr_queue_name(0), SVR_CONSUMER_GROUP_NAME
            )
            if group_info is not None:
                PENDING_TASKS = int(group_info.get("pending", 0))
                LAG_TASKS = int(group_info.get("lag", 0))

            current = copy.deepcopy(CURRENT_TASKS)
            heartbeat = json.dumps(
                {
                    "name": CONSUMER_NAME,
                    "now": now.astimezone().isoformat(timespec="milliseconds"),
                    "boot_at": BOOT_AT,
                    "pending": PENDING_TASKS,
                    "lag": LAG_TASKS,
                    "done": DONE_TASKS,
                    "failed": FAILED_TASKS,
                    "current": current,
                }
            )
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
                        consumer_name,
                        now.timestamp() - WORKER_HEARTBEAT_TIMEOUT,
                        now.timestamp() + 10,
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
    这是一个异步函数，被 nursery.start_soon 调用，在后台运行。主要作用是提供一个健壮的框架，用于调用 handle_task，并确保在任务完成后（或失败后）正确地释放并发控制的令牌。、
    task_manager()为什么要通过handle_task()调用do_handle_task()，而不是直接调用do_handle_task()？(分层和职责分离原则)
    这是因为，在 do_handle_task() 中，可能会发生异常，导致任务失败。但是，如果直接在 task_manager() 中调用 do_handle_task()，那么异常会直接导致任务失败，而无法被 handle_task() 捕获。
    因此，通过调用 handle_task()，可以在异常发生时，将异常信息传递给 handle_task()，并确保在任务完成后（或失败后）正确地释放并发控制的令牌。
    通过将 handle_task() 放在 try...finally 结构中，task_manager() 确保了无论任务处理成功、失败，甚至发生意想不到的崩溃，task_limiter.release() 这行代码总会被执行。
    如果没有这一层封装，task_manager() 的 try...finally 结构就必须变得复杂，它需要知道 handle_task 失败了，然后去处理 task_limiter 的释放。
    而现在，task_manager 只需要关心一件事情：在任务开始时拿到令牌，在任务结束时（无论何种方式）释放令牌。
    task_manager() 封装了并发控制的逻辑。
    handle_task() 封装了任务状态管理和异常处理的逻辑。
    do_handle_task() 封装了具体的业务逻辑。
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

    ## 第1步：初始化日志、版本信息和配置。
    # 打印一个 ASCII 艺术的 RAGFlow logo 到日志。
    logging.info(
        r"""
  ______           __      ______                     __
 /_  __/___ ______/ /__   / ____/  _____  _______  __/ /_____  _____
  / / / __ `/ ___/ //_/  / __/ | |/_/ _ \/ ___/ / / / __/ __ \/ ___/
 / / / /_/ (__  ) ,<    / /____>  </  __/ /__/ /_/ / /_/ /_/ / /
/_/  \__,_/____/_/|_|  /_____/_/|_|\___/\___/\__,_/\__/\____/_/
    """
    )

    # 打印版本信息，并初始化和打印配置设置。在服务启动时确认当前运行的版本和配置，是排查问题的第一步。确保配置正确，可以避免许多因配置错误导致的问题。
    # 调用函数获取版本号。
    logging.info(f"TaskExecutor: RAGFlow version: {get_ragflow_version()}")
    # 调用 settings 模块的初始化函数，通常用于从环境变量或配置文件加载和验证配置。
    settings.init_settings()
    # 打印出当前的 RAG 相关配置，以便于调试和确认。
    print_rag_settings()

    ## 第2步：设置信号处理，实现优雅退出和内存追踪。
    # 这段代码块用于配置和启动内存追踪工具 tracemalloc。tracemalloc 是 Python 标准库中一个强大的内存追踪工具，它能帮助开发者定位内存泄漏问题。通过信号动态控制，可以在不重启服务的情况下对内存使用情况进行诊断，这对于长时间运行的服务至关重要。
    # 判断操作系统是否为 Windows，因为 SIGUSR1 和 SIGUSR2 是 Unix-like 系统特有的信号。
    # SIGUSR1 信号会触发 start_tracemalloc_and_snapshot 函数，SIGUSR2 信号会触发 stop_tracemalloc。
    if sys.platform != "win32":
        signal.signal(signal.SIGUSR1, start_tracemalloc_and_snapshot)
        signal.signal(signal.SIGUSR2, stop_tracemalloc)
    # 从环境变量中获取 TRACE_MALLOC_ENABLED 的值。
    TRACE_MALLOC_ENABLED = int(os.environ.get("TRACE_MALLOC_ENABLED", "0"))
    # 如果该环境变量设置为 1，则在服务启动时立即开启内存追踪。
    if TRACE_MALLOC_ENABLED:
        start_tracemalloc_and_snapshot(None, None)

    # 这是为终止信号 SIGINT 和 SIGTERM 注册处理函数。
    # SIGINT 信号通常由 Ctrl+C 触发，SIGTERM 信号是 kill 命令发送的默认终止信号。
    # 这两行代码将这些信号与一个名为 signal_handler 的函数关联起来。当服务需要停止时，它能够执行一个预定义的 signal_handler 函数来清理资源、关闭连接，并优雅地退出，而不是立即被强制终止。
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ## 第3步：使用 trio.open_nursery() 启动一个并发上下文，并在其中开始两个核心异步任务：report_status（状态报告）和 task_manager（任务管理）。
    """==============================================================
    这是整个任务执行器的核心异步循环，负责启动状态报告report_status和任务管理task_manager。
    trio 框架中的一个核心概念，它创建了一个“育儿园”（nursery）。在这个 nursery 中启动的所有子任务（start_soon）都会被自动管理。当 nursery 退出时，它会确保所有子任务都已完成或被取消。
    为什么？
    结构化并发：trio 的 nursery 模型使得并发编程更加安全和直观。它可以自动处理子任务的生命周期和错误传播。
    资源管理和任务调度：report_status 确保了服务的可监控性。task_limiter 确保了任务执行器不会因为同时处理太多任务而过载。
    解耦与模块化：将状态报告、任务调度和任务处理逻辑分别封装在不同的异步函数中，使代码结构清晰，易于维护。
    代码执行流程总结：
    1.	trio.run(main) 被调用，事件循环开始，main 函数作为根任务被启动。
    2.	main 函数进入 async with trio.open_nursery() as nursery: 上下文。
    3.	并发任务 1：nursery.start_soon(report_status) 立即启动 report_status 任务，这个任务在后台独立运行，定期报告服务状态。
    4.	main 函数进入 while 循环。
    5.	main 阻塞在 await task_limiter.acquire() 这一行，直到有可用的任务槽位。
    6.	一旦获得一个令牌（例如，当一个旧的 task_manager 任务完成并释放了令牌），main 立即继续执行。
    7.	并发任务 2：nursery.start_soon(task_manager) 立即启动一个 task_manager 任务。这个新任务会去队列中拉取并处理一个具体的文档。
    8.	main 函数立即回到循环的开头，再次尝试 await task_limiter.acquire()，等待下一个任务槽位。
    9.	这个循环持续进行，不断地启动新的 task_manager 任务，直到：
    o	脚本接收到终止信号（SIGINT 或 SIGTERM）。
    o	signal_handler 函数被调用，并设置 stop_event。
    o	下一次循环条件检查 not stop_event.is_set() 变为 False，循环结束。
    10.	main 函数的 async with 块结束，nursery 会等待所有已启动的 task_manager 任务（包括那些正在处理中的）完成。
    11.	所有子任务完成后，trio.open_nursery() 退出，main 函数结束。
    12.	trio.run(main) 退出，程序终止。
    =============================================================="""
    # 这是一个 trio 框架中的特殊语法，称为“育儿园（Nursery）”。它创建了一个异步上下文管理器。
    # trio.open_nursery() 返回一个 nursery 对象，并将其绑定到 nursery 变量上。
    # 类比：一个幼儿园老师 nursery 开始接手管理一群孩子（异步任务）。
    async with trio.open_nursery() as nursery:

        # 【并发任务1】
        # 在后台启动一个名为 report_status 的异步任务，该任务可能定期向某个地方（如 Redis 或数据库）报告当前任务执行器的健康状态。
        # 类比：老师 nursery 让一个叫“报告员”的孩子（report_status 任务）去角落里玩。这个孩子每隔一段时间就会向老师汇报一下情况（比如“小明在玩玩具”，“小红在睡觉”）。
        nursery.start_soon(report_status)

        # 这是一个无限循环，只要 stop_event（一个异步事件对象）没有被触发，循环就会一直运行。
        # 当脚本接收到 SIGINT (Ctrl+C) 或 SIGTERM (kill) 信号时，signal_handler 函数会被调用(如上面两行代码)，而这个函数会设置 stop_event，从而中断循环。
        # 类比：老师 nursery 知道，她要一直工作，直到听到“放学铃”响了才结束。这个“放学铃”就是 stop_event。只要放学铃没响，她就继续工作。
        # 当放学铃响了（stop_event 被设置），while 循环就结束了。老师 nursery 准备下班。在老师下班前，她必须确保活动室里的所有孩子都已经被接走了。
        while not stop_event.is_set():

            # 这里使用了task_limiter，负责流量控制，确保不会同时启动过多的 task_manager 任务。
            # await 关键字会暂停当前 main 函数的执行，直到 task_limiter 有可用的令牌，允许启动新任务（限制同时运行的任务数量）。
            # 类比：老师 nursery 知道，她同时只能看管有限数量的孩子，以免忙不过来。所以，在接新孩子进来之前，她会先去查看一下“空位表”（task_limiter）。
            # 如果空位表显示已经满员了，老师就会停下来等，直到有孩子被接走了，腾出一个空位来。这对应了代码中的 await，暂停等待。
            await task_limiter.acquire()

            # 【并发任务2】
            # 当task_limiter 允许启动新任务时，它就立即启动一个 task_manager 的异步任务。task_manager 才是真正负责从队列中拉取和处理任务的核心函数。
            # 类比：一旦空位表有空位了，老师就会让一个新来的孩子，也就是“任务处理员”（task_manager 任务），进入活动室。
            nursery.start_soon(task_manager)

    # 一条错误日志，表示这是一个不应该被执行到的代码行。
    logging.error("BUG!!! You should not reach here!!!")


if __name__ == "__main__":
    # 这是一个 Python 标准库 faulthandler 中的函数调用。faulthandler 模块可以帮助诊断程序崩溃（例如，分段错误 segmentation fault）。
    # faulthandler.enable() 会注册一个处理程序，当 Python 解释器检测到致命错误时，它会打印一个 Python 回溯（traceback），包括所有正在运行的线程的栈信息。
    faulthandler.enable()

    # 这是一个日志初始化函数的调用，可能来自 api.utils.log_utils 模块。它的作用是配置程序的日志系统。
    init_root_logger(CONSUMER_NAME)

    """这是一个异步框架 trio 的入口点。它启动并运行整个异步应用程序。是整个任务执行器程序的起点。
    trio.run() 接收一个异步函数（main）作为参数。它会创建一个新的事件循环，并在该循环中执行 main 函数。
    这行代码本身不会循环执行，但它启动了 main 函数，而 main 函数内部是包含了循环的。
    类比：trio.run(main) 就像是按下了“启动”按钮，它开启了整个工厂的自动化流水线（事件循环），然后把 main 函数这个“总指挥官”送上流水线。
    main 函数会不断地循环执行它的任务，但它会通过 await 关键字，聪明地将控制权让给流水线上的其他工人，从而实现高效的并行工作。"""
    trio.run(main)
