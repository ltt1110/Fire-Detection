#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elasticsearch Service for Fire Detection
=========================================
Ghi log lịch sử phát hiện lửa vào Elasticsearch.
Nếu ES không sẵn sàng → tự động fallback sang JSON file.

Cách chạy Elasticsearch (Docker):
    docker-compose up -d          # tại thư mục PythonProject/
    http://localhost:9200         # ES API
    http://localhost:5601         # Kibana UI (optional)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Load biến môi trường từ .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv chưa cài, dùng os.environ trực tiếp

# Cấu hình từ .env
ES_HOST   = os.getenv("ES_HOST",  "localhost")
ES_PORT   = int(os.getenv("ES_PORT", "9200"))
ES_INDEX  = os.getenv("ES_INDEX", "fire-detection-events")
FALLBACK_LOG = os.path.join(os.path.dirname(__file__), "logs", "detection_history.jsonl")

logger = logging.getLogger(__name__)

# ── Thử import elasticsearch ────────────────────────────────
try:
    from elasticsearch import Elasticsearch
    _ES_PKG_OK = True
except ImportError:
    _ES_PKG_OK = False
    logger.warning("⚠️  Gói 'elasticsearch' chưa cài. Chạy: pip install elasticsearch")


# ════════════════════════════════════════════════════════════
class FireDetectionLogger:
    """
    Logger lịch sử phát hiện lửa.
    - Ưu tiên ghi vào Elasticsearch.
    - Fallback sang file JSONL nếu ES không khả dụng.
    """

    def __init__(self):
        self.es:       Optional[Any] = None
        self.use_es:   bool = False
        self._init_elasticsearch()
        self._init_fallback_file()

    # ── Khởi tạo ES ─────────────────────────────────────────
    def _init_elasticsearch(self):
        if not _ES_PKG_OK:
            return
        try:
            self.es = Elasticsearch(
                f"http://{ES_HOST}:{ES_PORT}",
                request_timeout=5,
                max_retries=1,
                retry_on_timeout=False,
            )
            if self.es.ping():
                self._ensure_index()
                self.use_es = True
                print(f"[ES] Connected -> {ES_HOST}:{ES_PORT}  index={ES_INDEX}")
            else:
                print(f"[ES] No response -> using JSON fallback ({FALLBACK_LOG})")
        except Exception as exc:
            print(f"[ES] Connection error: {exc} -> using JSON fallback")

    def _ensure_index(self):
        """Tạo ES index với mapping nếu chưa tồn tại."""
        if self.es.indices.exists(index=ES_INDEX):
            return
        self.es.indices.create(
            index=ES_INDEX,
            mappings={
                "properties": {
                    "timestamp":       {"type": "date"},
                    "source":          {"type": "keyword"},   # "upload" | "webcam"
                    "is_fire":         {"type": "boolean"},
                    "is_confirmed":    {"type": "boolean"},   # temporal confirmed
                    "location":        {"type": "keyword"},
                    "best_model_name": {"type": "keyword"},
                    "best_confidence": {"type": "float"},
                    "fire_votes":      {"type": "integer"},   # số model vote FIRE
                    "total_models":    {"type": "integer"},
                    "image_filename":  {"type": "keyword"},
                    # predictions object lưu nguyên, không index → tiết kiệm disk
                    "predictions":     {"type": "object", "enabled": False},
                }
            },
            settings={"number_of_shards": 1, "number_of_replicas": 0},
        )
        print(f"[ES] Created index: {ES_INDEX}")

    # ── Khởi tạo fallback file ───────────────────────────────
    def _init_fallback_file(self):
        os.makedirs(os.path.dirname(FALLBACK_LOG), exist_ok=True)
        if not os.path.exists(FALLBACK_LOG):
            open(FALLBACK_LOG, "w", encoding="utf-8").close()

    # ════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════

    def log_event(
        self,
        source: str,
        predictions: Dict,
        best_model: Dict,
        image_filename: str = None,
        is_confirmed: bool = True,
    ) -> bool:
        """
        Ghi một sự kiện phát hiện.

        Args:
            source:         "upload" hoặc "webcam"
            predictions:    dict kết quả từ tất cả models
            best_model:     {"name": ..., "prediction": ..., "confidence": ...}
            image_filename: tên file ảnh (với upload)
            is_confirmed:   với webcam, True khi đã qua temporal check
        """
        is_fire   = best_model.get("prediction") == "FIRE"
        fire_votes = sum(1 for p in predictions.values() if p.get("prediction") == "FIRE")

        event = {
            "timestamp":       datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "source":          source,
            "is_fire":         is_fire,
            "is_confirmed":    is_confirmed,
            "location":        os.getenv("CAMERA_LOCATION", "Camera AI"),
            "best_model_name": best_model.get("name", "Unknown"),
            "best_confidence": float(best_model.get("confidence", 0)),
            "fire_votes":      fire_votes,
            "total_models":    len(predictions),
            "image_filename":  image_filename,
            "predictions": {
                k: {
                    "prediction":       v.get("prediction"),
                    "confidence":       round(float(v.get("confidence", 0)), 4),
                    "probability_fire": round(float(v.get("probability_fire", 0)), 4),
                }
                for k, v in predictions.items()
            },
        }

        # Ghi vào ES
        if self.use_es:
            try:
                self.es.index(index=ES_INDEX, document=event)
                return True
            except Exception as exc:
                logger.error(f"[ES] Lỗi index: {exc}")

        # Fallback → JSONL file
        try:
            with open(FALLBACK_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            return True
        except Exception as exc:
            logger.error(f"[Fallback] Lỗi ghi file: {exc}")
            return False

    # ── Lấy lịch sử ─────────────────────────────────────────
    def get_recent_events(
        self,
        limit: int = 50,
        source: str = None,
        is_fire: bool = None,
        hours: int = 24,
    ) -> List[Dict]:
        if self.use_es:
            return self._es_recent(limit, source, is_fire, hours)
        return self._file_recent(limit, source, is_fire, hours)

    def _es_recent(self, limit, source, is_fire, hours) -> List[Dict]:
        must = [{"range": {"timestamp": {"gte": f"now-{hours}h"}}}]
        if source:
            must.append({"term": {"source": source}})
        if is_fire is not None:
            must.append({"term": {"is_fire": is_fire}})
        try:
            resp = self.es.search(
                index=ES_INDEX,
                query={"bool": {"must": must}},
                sort=[{"timestamp": {"order": "desc"}}],
                size=limit,
            )
            return [h["_source"] for h in resp["hits"]["hits"]]
        except Exception as exc:
            logger.error(f"[ES] Query lỗi: {exc}")
            return []

    def _file_recent(self, limit, source, is_fire, hours) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        events: List[Dict] = []
        try:
            with open(FALLBACK_LOG, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            for line in reversed(lines):
                if len(events) >= limit:
                    break
                try:
                    ev = json.loads(line.strip())
                    ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))
                    if ts.replace(tzinfo=None) < cutoff:
                        continue
                    if source and ev.get("source") != source:
                        continue
                    if is_fire is not None and ev.get("is_fire") != is_fire:
                        continue
                    events.append(ev)
                except Exception:
                    continue
        except FileNotFoundError:
            pass
        return events

    # ── Thống kê ─────────────────────────────────────────────
    def get_statistics(self, hours: int = 24) -> Dict:
        if self.use_es:
            return self._es_stats(hours)
        return self._file_stats(hours)

    def _es_stats(self, hours) -> Dict:
        try:
            resp = self.es.search(
                index=ES_INDEX,
                query={"range": {"timestamp": {"gte": f"now-{hours}h"}}},
                aggs={
                    "fire_count":    {"filter": {"term": {"is_fire": True}}},
                    "no_fire_count": {"filter": {"term": {"is_fire": False}}},
                    "by_source":     {"terms": {"field": "source"}},
                    "by_model":      {"terms": {"field": "best_model_name", "size": 10}},
                    "avg_confidence":{"avg":   {"field": "best_confidence"}},
                    "timeline": {
                        "date_histogram": {
                            "field": "timestamp",
                            "calendar_interval": "1h",
                            "min_doc_count": 0,
                            "extended_bounds": {
                                "min": f"now-{hours}h",
                                "max": "now",
                            },
                        }
                    },
                },
                size=0,
            )
            aggs  = resp["aggregations"]
            total = resp["hits"]["total"]["value"]
            return {
                "total_events":   total,
                "fire_count":     aggs["fire_count"]["doc_count"],
                "no_fire_count":  aggs["no_fire_count"]["doc_count"],
                "fire_rate":      round(aggs["fire_count"]["doc_count"] / max(total, 1), 4),
                "avg_confidence": round(float(aggs["avg_confidence"]["value"] or 0), 4),
                "by_source":  {b["key"]: b["doc_count"] for b in aggs["by_source"]["buckets"]},
                "by_model":   {b["key"]: b["doc_count"] for b in aggs["by_model"]["buckets"]},
                "timeline":   [
                    {"time": b["key_as_string"], "count": b["doc_count"]}
                    for b in aggs["timeline"]["buckets"]
                ],
                "data_source": "elasticsearch",
                "hours": hours,
            }
        except Exception as exc:
            logger.error(f"[ES] Stats lỗi: {exc}")
            return self._empty_stats(hours)

    def _file_stats(self, hours) -> Dict:
        events = self._file_recent(limit=100_000, source=None, is_fire=None, hours=hours)
        total  = len(events)
        fires  = sum(1 for e in events if e.get("is_fire"))
        by_src: Dict[str, int] = {}
        by_mdl: Dict[str, int] = {}
        for e in events:
            s = e.get("source", "unknown")
            m = e.get("best_model_name", "unknown")
            by_src[s] = by_src.get(s, 0) + 1
            by_mdl[m] = by_mdl.get(m, 0) + 1
        avg_conf = sum(e.get("best_confidence", 0) for e in events) / max(total, 1)
        return {
            "total_events":   total,
            "fire_count":     fires,
            "no_fire_count":  total - fires,
            "fire_rate":      round(fires / max(total, 1), 4),
            "avg_confidence": round(avg_conf, 4),
            "by_source":      by_src,
            "by_model":       by_mdl,
            "timeline":       [],   # chưa hỗ trợ trong file mode
            "data_source":    "json_file",
            "hours":          hours,
        }

    def _empty_stats(self, hours=24) -> Dict:
        return {
            "total_events": 0, "fire_count": 0, "no_fire_count": 0,
            "fire_rate": 0, "avg_confidence": 0,
            "by_source": {}, "by_model": {}, "timeline": [],
            "data_source": "error", "hours": hours,
        }

    # ── Trạng thái ───────────────────────────────────────────
    @property
    def status(self) -> Dict:
        return {
            "es_package_installed": _ES_PKG_OK,
            "es_connected":         self.use_es,
            "es_endpoint":          f"http://{ES_HOST}:{ES_PORT}",
            "es_index":             ES_INDEX,
            "fallback_file":        FALLBACK_LOG,
        }


# ── Singleton ────────────────────────────────────────────────
_instance: Optional[FireDetectionLogger] = None


def get_logger() -> FireDetectionLogger:
    """Trả về singleton logger instance."""
    global _instance
    if _instance is None:
        _instance = FireDetectionLogger()
    return _instance
