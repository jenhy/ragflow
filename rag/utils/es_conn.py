#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
#

import logging
import re
import json
import time
import os

import copy
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch_dsl import UpdateByQuery, Q, Search, Index
from elastic_transport import ConnectionTimeout
from rag import settings
from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import singleton, get_float
from api.utils.file_utils import get_project_base_directory
from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    OrderByExpr,
    MatchTextExpr,
    MatchDenseExpr,
    FusionExpr,
)
from rag.nlp import is_english, rag_tokenizer

ATTEMPT_TIME = 2

logger = logging.getLogger("ragflow.es_conn")


@singleton
class ESConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use Elasticsearch {settings.ES['hosts']} as the doc engine.")
        for _ in range(ATTEMPT_TIME):
            try:
                if self._connect():
                    break
            except Exception as e:
                logger.warning(
                    f"{str(e)}. Waiting Elasticsearch {settings.ES['hosts']} to be healthy."
                )
                time.sleep(5)

        if not self.es.ping():
            msg = f"Elasticsearch {settings.ES['hosts']} is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)
        v = self.info.get("version", {"number": "8.11.3"})
        v = v["number"].split(".")[0]
        if int(v) < 8:
            msg = f"Elasticsearch version must be greater than or equal to 8, current version: {v}"
            logger.error(msg)
            raise Exception(msg)
        fp_mapping = os.path.join(get_project_base_directory(), "conf", "mapping.json")
        if not os.path.exists(fp_mapping):
            msg = f"Elasticsearch mapping file not found at {fp_mapping}"
            logger.error(msg)
            raise Exception(msg)
        self.mapping = json.load(open(fp_mapping, "r"))
        logger.info(f"Elasticsearch {settings.ES['hosts']} is healthy.")

    def _connect(self):
        self.es = Elasticsearch(
            settings.ES["hosts"].split(","),
            basic_auth=(
                (settings.ES["username"], settings.ES["password"])
                if "username" in settings.ES and "password" in settings.ES
                else None
            ),
            verify_certs=False,
            timeout=600,
        )
        if self.es:
            self.info = self.es.info()
            return True
        return False

    """
    Database operations
    """

    def dbType(self) -> str:
        return "elasticsearch"

    def health(self) -> dict:
        health_dict = dict(self.es.cluster.health())
        health_dict["type"] = "elasticsearch"
        return health_dict

    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        if self.indexExist(indexName, knowledgebaseId):
            return True
        try:
            from elasticsearch.client import IndicesClient

            return IndicesClient(self.es).create(
                index=indexName,
                settings=self.mapping["settings"],
                mappings=self.mapping["mappings"],
            )
        except Exception:
            logger.exception("ESConnection.createIndex error %s" % (indexName))

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        if len(knowledgebaseId) > 0:
            # The index need to be alive after any kb deletion since all kb under this tenant are in one index.
            return
        try:
            self.es.indices.delete(index=indexName, allow_no_indices=True)
        except NotFoundError:
            pass
        except Exception:
            logger.exception("ESConnection.deleteIdx error %s" % (indexName))

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        s = Index(indexName, self.es)
        for i in range(ATTEMPT_TIME):
            try:
                return s.exists()
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                logger.exception(e)
                break
        return False

    """
    CRUD operations
    """

    def search(
        self,
        selectFields: list[str],
        highlightFields: list[str],
        condition: dict,
        matchExprs: list[MatchExpr],
        orderBy: OrderByExpr,
        offset: int,
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        aggFields: list[str] = [],
        rank_feature: dict | None = None,
    ):
        """
        Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
        """
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0
        assert "_id" not in condition

        bqry = Q("bool", must=[])
        condition["kb_id"] = knowledgebaseIds
        for k, v in condition.items():
            if k == "available_int":
                if v == 0:
                    bqry.filter.append(Q("range", available_int={"lt": 1}))
                else:
                    bqry.filter.append(
                        Q("bool", must_not=Q("range", available_int={"lt": 1}))
                    )
                continue
            if not v:
                continue
            if isinstance(v, list):
                bqry.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bqry.filter.append(Q("term", **{k: v}))
            else:
                raise Exception(
                    f"Condition `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str or list."
                )

        s = Search()
        vector_similarity_weight = 0.5
        for m in matchExprs:
            if (
                isinstance(m, FusionExpr)
                and m.method == "weighted_sum"
                and "weights" in m.fusion_params
            ):
                assert (
                    len(matchExprs) == 3
                    and isinstance(matchExprs[0], MatchTextExpr)
                    and isinstance(matchExprs[1], MatchDenseExpr)
                    and isinstance(matchExprs[2], FusionExpr)
                )
                weights = m.fusion_params["weights"]
                vector_similarity_weight = get_float(weights.split(",")[1])
        for m in matchExprs:
            if isinstance(m, MatchTextExpr):
                minimum_should_match = m.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    minimum_should_match = str(int(minimum_should_match * 100)) + "%"
                bqry.must.append(
                    Q(
                        "query_string",
                        fields=m.fields,
                        type="best_fields",
                        query=m.matching_text,
                        minimum_should_match=minimum_should_match,
                        boost=1,
                    )
                )
                bqry.boost = 1.0 - vector_similarity_weight

            elif isinstance(m, MatchDenseExpr):
                assert bqry is not None
                similarity = 0.0
                if "similarity" in m.extra_options:
                    similarity = m.extra_options["similarity"]
                s = s.knn(
                    m.vector_column_name,
                    m.topn,
                    m.topn * 2,
                    query_vector=list(m.embedding_data),
                    filter=bqry.to_dict(),
                    similarity=similarity,
                )

        if bqry and rank_feature:
            for fld, sc in rank_feature.items():
                if fld != PAGERANK_FLD:
                    fld = f"{TAG_FLD}.{fld}"
                bqry.should.append(Q("rank_feature", field=fld, linear={}, boost=sc))

        if bqry:
            s = s.query(bqry)
        for field in highlightFields:
            s = s.highlight(field)

        if orderBy:
            orders = list()
            for field, order in orderBy.fields:
                order = "asc" if order == 0 else "desc"
                if field in ["page_num_int", "top_int"]:
                    order_info = {
                        "order": order,
                        "unmapped_type": "float",
                        "mode": "avg",
                        "numeric_type": "double",
                    }
                elif field.endswith("_int") or field.endswith("_flt"):
                    order_info = {"order": order, "unmapped_type": "float"}
                else:
                    order_info = {"order": order, "unmapped_type": "text"}
                orders.append({field: order_info})
            s = s.sort(*orders)

        for fld in aggFields:
            s.aggs.bucket(f"aggs_{fld}", "terms", field=fld, size=1000000)

        if limit > 0:
            s = s[offset : offset + limit]
        q = s.to_dict()
        logger.debug(f"ESConnection.search {str(indexNames)} query: " + json.dumps(q))

        for i in range(ATTEMPT_TIME):
            try:
                # print(json.dumps(q, ensure_ascii=False))
                res = self.es.search(
                    index=indexNames,
                    body=q,
                    timeout="600s",
                    # search_type="dfs_query_then_fetch",
                    track_total_hits=True,
                    _source=True,
                )
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                logger.debug(f"ESConnection.search {str(indexNames)} res: " + str(res))
                return res
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                self._connect()
                continue
            except Exception as e:
                logger.exception(
                    f"ESConnection.search {str(indexNames)} query: " + str(q) + str(e)
                )
                raise e

        logger.error(f"ESConnection.search timeout for {ATTEMPT_TIME} times!")
        raise Exception("ESConnection.search timeout.")

    def get(
        self, chunkId: str, indexName: str, knowledgebaseIds: list[str]
    ) -> dict | None:
        for i in range(ATTEMPT_TIME):
            try:
                res = self.es.get(
                    index=(indexName),
                    id=chunkId,
                    source=True,
                )
                if str(res.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                chunk = res["_source"]
                chunk["id"] = chunkId
                return chunk
            except NotFoundError:
                return None
            except Exception as e:
                logger.exception(f"ESConnection.get({chunkId}) got exception")
                raise e
        logger.error(f"ESConnection.get timeout for {ATTEMPT_TIME} times!")
        raise Exception("ESConnection.get timeout.")

    def insert(
        self, documents: list[dict], indexName: str, knowledgebaseId: str = None
    ) -> list[str]:
        # Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        """
        功能：如何将 RAGFlow 处理好的文档数据批量写入 Elasticsearch。
        参数：
        self: 表示 ESConnection 类的实例，其中包含了与 Elasticsearch 建立连接的对象 self.es。
        documents: list[dict]，文档数据列表。它是一个包含多个字典的列表，每个字典就是 embedding 函数返回的、带有向量和元数据的文档块。
        indexName: str，索引名称。要插入数据的 Elasticsearch 索引名称。
        knowledgebaseId: str，知识库的唯一 ID。
        返回：
        函数返回一个字符串列表，通常用于存放插入过程中遇到的错误信息。
        """

        ## 第1步：准备批量操作的数据结构。这段代码遍历输入的 documents 列表，为每个文档块构建一个 Elasticsearch 批量操作（Bulk API） 的请求体。
        # 初始化一个空列表，用于存放批量操作的请求体。
        operations = []
        for d in documents:
            # 这是一个安全检查，确保传入的字典中没有 Elasticsearch 内部使用的 _id 字段，但必须有 id 字段（这是 RAGFlow 为每个文档块生成的唯一 ID）。
            assert "_id" not in d
            assert "id" in d

            # 创建一个文档块的深拷贝，以避免修改原始数据。
            d_copy = copy.deepcopy(d)

            # 为每个文档块添加 kb_id 字段，这对于后续按知识库过滤检索非常重要。
            d_copy["kb_id"] = knowledgebaseId

            # 取出 id 字段，并将其从文档块的拷贝中移除。
            meta_id = d_copy.pop("id", "")

            # 这是批量操作的关键。Elasticsearch 的批量操作请求体格式是操作行和数据行交替出现的。
            # 第一个 append 插入一个操作行，告诉 Elasticsearch 在哪个索引 (_index) 下，以哪个 ID (_id) 执行 index 操作。
            operations.append({"index": {"_index": indexName, "_id": meta_id}})
            # 第二个 append 插入数据行，即文档块的实际内容，包括了原始文本、元数据和最重要的向量数据。
            operations.append(d_copy)

        ## 第2步：批量写入与重试机制。这段代码是真正执行批量写入操作的部分，并内置了重试机制。
        res = []
        # 一个循环，ATTEMPT_TIME 是一个常量，表示最大重试次数。这可以防止临时的网络波动导致任务失败。
        for _ in range(ATTEMPT_TIME):
            try:
                res = []
                # 调用 Elasticsearch 客户端库的 bulk 方法。
                # index=(indexName)：指定要操作的索引。
                # operations=operations：传入第二步构建好的批量操作列表。
                # refresh=False：设置为 False，意味着数据写入后不会立即刷新索引，以提高写入性能。刷新操作可以在后台异步进行。
                # timeout="60s"：设置请求的超时时间为 60 秒。
                r = self.es.bulk(
                    index=(indexName),
                    operations=operations,
                    refresh=False,
                    timeout="60s",
                )

                ## 第3步：结果处理与错误返回。 这段代码检查批量操作的执行结果，并收集任何发生的错误。
                # r["errors"] 是 Elasticsearch 返回结果中的一个布尔值，如果为 False，表示所有操作都成功。
                # 如果 r["errors"] 为 True，代码会遍历 r["items"] 列表，该列表包含了每个操作的详细结果。
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res
                # 它检查每个操作项中是否有 "error" 字段，如果有，就将该操作的 ID 和错误信息拼接到 res 列表中。
                # 为什么？
                # 即使批量操作成功返回，也不代表所有子操作都成功了。通过检查 items 中的错误，可以确保每个文档块都成功写入。
                # 将具体的错误信息返回，可以帮助上层调用者（do_handle_task）更好地记录和处理失败。
                for item in r["items"]:
                    for action in ["create", "delete", "index", "update"]:
                        if action in item and "error" in item[action]:
                            res.append(
                                str(item[action]["_id"])
                                + ":"
                                + str(item[action]["error"])
                            )
                return res
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                res.append(str(e))
                logger.warning("ESConnection.insert got exception: " + str(e))

        return res

    def update(
        self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str
    ) -> bool:
        doc = copy.deepcopy(newValue)
        doc.pop("id", None)
        condition["kb_id"] = knowledgebaseId
        if "id" in condition and isinstance(condition["id"], str):
            # update specific single document
            chunkId = condition["id"]
            for i in range(ATTEMPT_TIME):
                for k in doc.keys():
                    if "feas" != k.split("_")[-1]:
                        continue
                    try:
                        self.es.update(
                            index=indexName,
                            id=chunkId,
                            script=f'ctx._source.remove("{k}");',
                        )
                    except Exception:
                        logger.exception(
                            f"ESConnection.update(index={indexName}, id={chunkId}, doc={json.dumps(condition, ensure_ascii=False)}) got exception"
                        )
                try:
                    self.es.update(index=indexName, id=chunkId, doc=doc)
                    return True
                except Exception as e:
                    logger.exception(
                        f"ESConnection.update(index={indexName}, id={chunkId}, doc={json.dumps(condition, ensure_ascii=False)}) got exception: "
                        + str(e)
                    )
                    break
            return False

        # update unspecific maybe-multiple documents
        bqry = Q("bool")
        for k, v in condition.items():
            if not isinstance(k, str) or not v:
                continue
            if k == "exists":
                bqry.filter.append(Q("exists", field=v))
                continue
            if isinstance(v, list):
                bqry.filter.append(Q("terms", **{k: v}))
            elif isinstance(v, str) or isinstance(v, int):
                bqry.filter.append(Q("term", **{k: v}))
            else:
                raise Exception(
                    f"Condition `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str or list."
                )
        scripts = []
        params = {}
        for k, v in newValue.items():
            if k == "remove":
                if isinstance(v, str):
                    scripts.append(f"ctx._source.remove('{v}');")
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        scripts.append(
                            f"int i=ctx._source.{kk}.indexOf(params.p_{kk});ctx._source.{kk}.remove(i);"
                        )
                        params[f"p_{kk}"] = vv
                continue
            if k == "add":
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        scripts.append(f"ctx._source.{kk}.add(params.pp_{kk});")
                        params[f"pp_{kk}"] = vv.strip()
                continue
            if (not isinstance(k, str) or not v) and k != "available_int":
                continue
            if isinstance(v, str):
                v = re.sub(r"(['\n\r]|\\.)", " ", v)
                params[f"pp_{k}"] = v
                scripts.append(f"ctx._source.{k}=params.pp_{k};")
            elif isinstance(v, int) or isinstance(v, float):
                scripts.append(f"ctx._source.{k}={v};")
            elif isinstance(v, list):
                scripts.append(f"ctx._source.{k}=params.pp_{k};")
                params[f"pp_{k}"] = json.dumps(v, ensure_ascii=False)
            else:
                raise Exception(
                    f"newValue `{str(k)}={str(v)}` value type is {str(type(v))}, expected to be int, str."
                )
        ubq = UpdateByQuery(index=indexName).using(self.es).query(bqry)
        ubq = ubq.script(source="".join(scripts), params=params)
        ubq = ubq.params(refresh=True)
        ubq = ubq.params(slices=5)
        ubq = ubq.params(conflicts="proceed")

        for _ in range(ATTEMPT_TIME):
            try:
                _ = ubq.execute()
                return True
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                logger.error(
                    "ESConnection.update got exception: " + str(e) + "\n".join(scripts)
                )
                break
        return False

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        qry = None
        assert "_id" not in condition
        condition["kb_id"] = knowledgebaseId
        if "id" in condition:
            chunk_ids = condition["id"]
            if not isinstance(chunk_ids, list):
                chunk_ids = [chunk_ids]
            if not chunk_ids:  # when chunk_ids is empty, delete all
                qry = Q("match_all")
            else:
                qry = Q("ids", values=chunk_ids)
        else:
            qry = Q("bool")
            for k, v in condition.items():
                if k == "exists":
                    qry.filter.append(Q("exists", field=v))

                elif k == "must_not":
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if kk == "exists":
                                qry.must_not.append(Q("exists", field=vv))

                elif isinstance(v, list):
                    qry.must.append(Q("terms", **{k: v}))
                elif isinstance(v, str) or isinstance(v, int):
                    qry.must.append(Q("term", **{k: v}))
                else:
                    raise Exception("Condition value must be int, str or list.")
        logger.debug("ESConnection.delete query: " + json.dumps(qry.to_dict()))
        for _ in range(ATTEMPT_TIME):
            try:
                res = self.es.delete_by_query(
                    index=indexName, body=Search().query(qry).to_dict(), refresh=True
                )
                return res["deleted"]
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception as e:
                logger.warning("ESConnection.delete got exception: " + str(e))
                if re.search(r"(not_found)", str(e), re.IGNORECASE):
                    return 0
        return 0

    """
    Helper functions for search result
    """

    def getTotal(self, res):
        if isinstance(res["hits"]["total"], type({})):
            return res["hits"]["total"]["value"]
        return res["hits"]["total"]

    def getChunkIds(self, res):
        return [d["_id"] for d in res["hits"]["hits"]]

    def __getSource(self, res):
        rr = []
        for d in res["hits"]["hits"]:
            d["_source"]["id"] = d["_id"]
            d["_source"]["_score"] = d["_score"]
            rr.append(d["_source"])
        return rr

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        res_fields = {}
        if not fields:
            return {}
        for d in self.__getSource(res):
            m = {n: d.get(n) for n in fields if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, list):
                    m[n] = v
                    continue
                if n == "available_int" and isinstance(v, (int, float)):
                    m[n] = v
                    continue
                if not isinstance(v, str):
                    m[n] = str(m[n])
                # if n.find("tks") > 0:
                #     m[n] = rmSpace(m[n])

            if m:
                res_fields[d["id"]] = m
        return res_fields

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        ans = {}
        for d in res["hits"]["hits"]:
            hlts = d.get("highlight")
            if not hlts:
                continue
            txt = "...".join([a for a in list(hlts.items())[0][1]])
            if not is_english(txt.split()):
                ans[d["_id"]] = txt
                continue

            txt = d["_source"][fieldnm]
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            for t in re.split(r"[.?!;\n]", txt):
                for w in keywords:
                    t = re.sub(
                        r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])"
                        % re.escape(w),
                        r"\1<em>\2</em>\3",
                        t,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                if not re.search(
                    r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE
                ):
                    continue
                txts.append(t)
            ans[d["_id"]] = (
                "...".join(txts)
                if txts
                else "...".join([a for a in list(hlts.items())[0][1]])
            )

        return ans

    def getAggregation(self, res, fieldnm: str):
        agg_field = "aggs_" + fieldnm
        if "aggregations" not in res or agg_field not in res["aggregations"]:
            return list()
        bkts = res["aggregations"][agg_field]["buckets"]
        return [(b["key"], b["doc_count"]) for b in bkts]

    """
    SQL
    """

    def sql(self, sql: str, fetch_size: int, format: str):
        logger.debug(f"ESConnection.sql get sql: {sql}")
        sql = re.sub(r"[ `]+", " ", sql)
        sql = sql.replace("%", "")
        replaces = []
        for r in re.finditer(r" ([a-z_]+_l?tks)( like | ?= ?)'([^']+)'", sql):
            fld, v = r.group(1), r.group(3)
            match = " MATCH({}, '{}', 'operator=OR;minimum_should_match=30%') ".format(
                fld, rag_tokenizer.fine_grained_tokenize(rag_tokenizer.tokenize(v))
            )
            replaces.append(
                ("{}{}'{}'".format(r.group(1), r.group(2), r.group(3)), match)
            )

        for p, r in replaces:
            sql = sql.replace(p, r, 1)
        logger.debug(f"ESConnection.sql to es: {sql}")

        for i in range(ATTEMPT_TIME):
            try:
                res = self.es.sql.query(
                    body={"query": sql, "fetch_size": fetch_size},
                    format=format,
                    request_timeout="2s",
                )
                return res
            except ConnectionTimeout:
                logger.exception("ES request timeout")
                time.sleep(3)
                self._connect()
                continue
            except Exception:
                logger.exception("ESConnection.sql got exception")
                break
        logger.error(f"ESConnection.sql timeout for {ATTEMPT_TIME} times!")
        return None
