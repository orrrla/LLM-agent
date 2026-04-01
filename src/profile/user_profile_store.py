# -*- coding: utf-8 -*-
import json
import os
import threading
from datetime import datetime


DEFAULT_STORE_PATH = "./data/user_profile_store.json"


class UserProfileStore:
    def __init__(self, store_path=DEFAULT_STORE_PATH, memory_window=5):
        self.store_path = store_path
        self.memory_window = memory_window
        self._lock = threading.Lock()
        self._bootstrap()

    def _bootstrap(self):
        parent_dir = os.path.dirname(self.store_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, "w", encoding="utf-8") as fw:
                fw.write(json.dumps({}, ensure_ascii=False, indent=2))

    def _read(self):
        with open(self.store_path, "r", encoding="utf-8") as fd:
            payload = fd.read().strip()
            if not payload:
                return {}
            return json.loads(payload)

    def _write(self, data):
        with open(self.store_path, "w", encoding="utf-8") as fw:
            fw.write(json.dumps(data, ensure_ascii=False, indent=2))

    def get_profile(self, user_id):
        data = self._read()
        user_data = data.get(user_id, {})
        return user_data.get("profile", {})

    def upsert_profile(self, user_id, model_cfg="", software_version=""):
        with self._lock:
            data = self._read()
            user_data = data.get(user_id, {})
            profile = user_data.get("profile", {})
            if model_cfg:
                profile["model_cfg"] = model_cfg.strip()
            if software_version:
                profile["software_version"] = software_version.strip()
            profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
            user_data["profile"] = profile
            data[user_id] = user_data
            self._write(data)
            return profile

    def get_recent_turns(self, user_id):
        data = self._read()
        user_data = data.get(user_id, {})
        turns = user_data.get("recent_turns", [])
        return turns[-self.memory_window :]

    def append_turn(self, user_id, query, answer):
        with self._lock:
            data = self._read()
            user_data = data.get(user_id, {})
            turns = user_data.get("recent_turns", [])
            turns.append(
                {
                    "query": query.strip(),
                    "answer": answer.strip(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            user_data["recent_turns"] = turns[-self.memory_window :]
            data[user_id] = user_data
            self._write(data)

