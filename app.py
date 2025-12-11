"""
Gemini File Search Store Manager & RAG Chat (Streamlit App)
===========================================================
Fixed Version: Footer Export, History Manager, Full Cleanup
"""

import contextlib
import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import (
  Integer,
  String,
  Text,
  create_engine,
  delete,
  desc,
  func,
  select,
  update,
)
from sqlalchemy.orm import (
  DeclarativeBase,
  Mapped,
  Session,
  mapped_column,
  sessionmaker,
)

# --- Google GenAI SDK ---
try:
  from google import genai
  from google.genai import types
except ImportError:
  st.error("google-genai SDK not found. Please install: uv add google-genai")
  st.stop()

# --- Configuration ---
load_dotenv()

# Ensure strictly str
DEFAULT_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_RETRIES = 3

DB_URL = "sqlite:///gemini_store.db"
INPUT_DIR = Path("input_files")
INPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeminiApp")


# ==========================================
# 1. Database Schema
# ==========================================


class Base(DeclarativeBase):
  pass


class StoreRecord(Base):
  __tablename__ = "store_records"
  id: Mapped[str] = mapped_column(String(255), primary_key=True)
  display_name: Mapped[str] = mapped_column(String(255))
  status: Mapped[str] = mapped_column(String(20), default="active")
  created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class FileRecord(Base):
  __tablename__ = "file_records"
  id: Mapped[str] = mapped_column(
    String(36),
    primary_key=True,
    default=lambda: str(uuid.uuid4()),
  )
  store_id: Mapped[str] = mapped_column(String(255), index=True)
  file_name: Mapped[str] = mapped_column(String(255))
  local_path: Mapped[str] = mapped_column(String(1024))
  md5: Mapped[str] = mapped_column(String(32))
  file_size: Mapped[int] = mapped_column(Integer)
  suffix: Mapped[str | None] = mapped_column(String(20), nullable=True)
  status: Mapped[str] = mapped_column(String(20), default="active")
  created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class ChatSession(Base):
  __tablename__ = "chat_sessions"
  id: Mapped[str] = mapped_column(
    String(36),
    primary_key=True,
    default=lambda: str(uuid.uuid4()),
  )
  name: Mapped[str] = mapped_column(String(255))
  store_id: Mapped[str] = mapped_column(String(255))
  created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class ChatMessage(Base):
  __tablename__ = "chat_messages"
  id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
  session_id: Mapped[str] = mapped_column(String(36), index=True)
  role: Mapped[str] = mapped_column(String(20))
  content: Mapped[str] = mapped_column(Text)
  citations: Mapped[str | None] = mapped_column(Text, nullable=True)
  created_at: Mapped[datetime] = mapped_column(default=datetime.now)


# ==========================================
# 2. Logic Manager
# ==========================================


class GeminiManager:
  def __init__(self, db_url: str):
    self.engine = create_engine(db_url)
    Base.metadata.create_all(self.engine)
    self.SessionLocal = sessionmaker(bind=self.engine)
    self._sync_legacy_stores()

  def get_client(self, api_key: str):
    return genai.Client(api_key=api_key)

  def get_session(self) -> Session:
    return self.SessionLocal()

  def _sync_legacy_stores(self):
    with self.get_session() as session:
      stmt = (
        select(FileRecord.store_id)
        .outerjoin(StoreRecord, FileRecord.store_id == StoreRecord.id)
        .where(FileRecord.status == "active", StoreRecord.id.is_(None))
        .distinct()
      )
      orphan_ids = session.scalars(stmt).all()
      for oid in orphan_ids:
        if not oid:
          continue
        rec = StoreRecord(
          id=oid,
          display_name=f"Imported {oid.split('/')[-1]}",
          status="active",
        )
        session.add(rec)
      if orphan_ids:
        session.commit()

  def save_physical_file(self, uploaded_file) -> tuple[Path, str]:
    file_bytes = uploaded_file.getvalue()
    # S324: MD5 used for change detection
    md5 = hashlib.md5(file_bytes, usedforsecurity=False).hexdigest()
    ext = Path(uploaded_file.name).suffix
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = INPUT_DIR / safe_name
    with save_path.open("wb") as f:
      f.write(file_bytes)
    return save_path, md5

  def list_active_stores(self) -> list[dict]:
    with self.get_session() as session:
      stores = session.scalars(
        select(StoreRecord).where(StoreRecord.status == "active").order_by(desc(StoreRecord.created_at)),
      ).all()
      result = []
      for s in stores:
        count = session.scalar(
          select(func.count(FileRecord.id)).where(
            FileRecord.store_id == s.id,
            FileRecord.status == "active",
          ),
        )
        result.append(
          {
            "store_id": s.id,
            "display_name": s.display_name,
            "created_at": s.created_at,
            "file_count": count or 0,
          },
        )
      return result

  def get_store_files(self, store_id: str) -> list[dict]:
    with self.get_session() as session:
      stmt = select(FileRecord).where(
        FileRecord.store_id == store_id,
        FileRecord.status == "active",
      )
      return [
        {
          "file_name": r.file_name,
          "local_path": r.local_path,
          "size": r.file_size,
          "md5": r.md5,
          "created_at": r.created_at,
        }
        for r in session.scalars(stmt).all()
      ]

  def create_store(self, client: genai.Client, display_name: str) -> str:
    store = client.file_search_stores.create(config={"display_name": display_name})
    if not store.name:
      msg = "API returned no store name"
      raise ValueError(msg)
    with self.get_session() as session:
      session.add(
        StoreRecord(id=store.name, display_name=display_name, status="active"),
      )
      session.commit()
    return store.name

  def delete_store(self, client: genai.Client, store_id: str):
    """Delete store from API, DB, and Local Filesystem."""
    # 1. Delete from API
    try:
      client.file_search_stores.delete(name=store_id)
    except Exception as e:  # noqa: BLE001
      logger.warning(f"API delete warning: {e}")

    # 2. Delete Physical Files
    with self.get_session() as session:
      # Find all active files associated with this store
      files = session.scalars(
        select(FileRecord).where(
          FileRecord.store_id == store_id,
          FileRecord.status == "active",
        ),
      ).all()

      for f in files:
        try:
          p = Path(f.local_path)
          p.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
          logger.warning(f"Failed to delete local file {f.local_path}: {e}")

      # 3. Logical Delete in DB
      session.execute(
        update(StoreRecord).where(StoreRecord.id == store_id).values(status="deleted"),
      )
      session.execute(
        update(FileRecord).where(FileRecord.store_id == store_id).values(status="deleted"),
      )
      session.commit()

  def _upload_single_file(
    self,
    session: Session,
    client: genai.Client,
    store_id: str,
    up_file: Any,
    saved_path: Path,
    md5: str,
  ) -> bool:
    for attempt in range(MAX_RETRIES):
      try:
        op = client.file_search_stores.upload_to_file_search_store(
          file=str(saved_path),
          file_search_store_name=store_id,
          config={"display_name": up_file.name},
        )
        while not getattr(op, "done", False):
          time.sleep(1)
          op = client.operations.get(op)

        session.add(
          FileRecord(
            store_id=store_id,
            file_name=up_file.name,
            local_path=str(saved_path),
            md5=md5,
            file_size=saved_path.stat().st_size,
            suffix=saved_path.suffix,
            status="active",
          ),
        )
        session.commit()
      except Exception as e:  # noqa: BLE001
        logger.warning(f"Retry {attempt + 1}: {e}")
        if attempt < (MAX_RETRIES - 1):
          time.sleep((2**attempt) + random.uniform(0, 1))  # noqa: S311
      else:
        return True
    return False

  def upload_files(
    self,
    client: genai.Client,
    store_id: str,
    uploaded_files: list[Any],
    progress_bar=None,
  ) -> dict[str, list[str]]:
    total = len(uploaded_files)
    results: dict[str, list[str]] = {
      "uploaded": [],
      "skipped": [],
      "failed": [],
    }

    with self.get_session() as session:
      for idx, up_file in enumerate(uploaded_files):
        if progress_bar:
          progress_bar.progress(
            (idx) / total,
            text=f"Processing {up_file.name}...",
          )
        try:
          saved_path, md5 = self.save_physical_file(up_file)
          existing = session.scalar(
            select(FileRecord).where(
              FileRecord.store_id == store_id,
              FileRecord.file_name == up_file.name,
              FileRecord.status == "active",
            ),
          )
          if existing and existing.md5 == md5:
            results["skipped"].append(up_file.name)
            with contextlib.suppress(Exception):
              saved_path.unlink()
            continue

          if self._upload_single_file(
            session,
            client,
            store_id,
            up_file,
            saved_path,
            md5,
          ):
            results["uploaded"].append(up_file.name)
          else:
            results["failed"].append(up_file.name)

          time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
          logger.exception(f"Critical error processing {up_file.name}: {e}")  # noqa: TRY401
          results["failed"].append(up_file.name)

      if progress_bar:
        progress_bar.progress(1.0, text="Done!")
    return results

  def refresh_store(self, client: genai.Client, old_store_id: str) -> str:
    with self.get_session() as session:
      old = session.scalar(
        select(StoreRecord).where(StoreRecord.id == old_store_id),
      )
      base = old.display_name if old else "refreshed"
    new_id = self.create_store(client, f"{base}-{int(time.time())}")

    with self.get_session() as session:
      records = session.scalars(
        select(FileRecord).where(
          FileRecord.store_id == old_store_id,
          FileRecord.status == "active",
        ),
      ).all()
      for rec in records:
        p = Path(rec.local_path)
        if not p.exists():
          continue
        with contextlib.suppress(Exception):
          op = client.file_search_stores.upload_to_file_search_store(
            file=str(p),
            file_search_store_name=new_id,
            config={"display_name": rec.file_name},
          )
          while not getattr(op, "done", False):
            time.sleep(1)
            op = client.operations.get(op)
          session.add(
            FileRecord(
              store_id=new_id,
              file_name=rec.file_name,
              local_path=rec.local_path,
              md5=rec.md5,
              file_size=rec.file_size,
              suffix=rec.suffix,
              status="active",
            ),
          )
          time.sleep(1.0)
      session.commit()
    self.delete_store(client, old_store_id)
    return new_id

  # --- Chat Logic ---
  def create_chat_session(self, name: str, store_id: str) -> str:
    with self.get_session() as session:
      sess = ChatSession(name=name, store_id=store_id)
      session.add(sess)
      session.commit()
      return sess.id

  def delete_chat_session(self, session_id: str):
    with self.get_session() as session:
      session.execute(delete(ChatSession).where(ChatSession.id == session_id))
      session.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
      session.commit()

  def update_session_name(self, session_id: str, new_name: str):
    with self.get_session() as session:
      session.execute(
        update(ChatSession).where(ChatSession.id == session_id).values(name=new_name),
      )
      session.commit()

  def list_chat_sessions(self) -> list[dict]:
    with self.get_session() as session:
      stmt = select(ChatSession).order_by(desc(ChatSession.created_at))
      return [{"id": s.id, "name": s.name} for s in session.scalars(stmt).all()]

  def get_chat_history(self, session_id: str) -> list[dict]:
    with self.get_session() as session:
      stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
      return [
        {
          "role": m.role,
          "content": m.content,
          "citations": json.loads(m.citations) if m.citations else [],
        }
        for m in session.scalars(stmt).all()
      ]

  def save_message(
    self,
    session_id: str,
    role: str,
    content: str,
    citations: list[str] | None = None,
  ):
    with self.get_session() as session:
      session.add(
        ChatMessage(
          session_id=session_id,
          role=role,
          content=content,
          citations=json.dumps(citations, ensure_ascii=False) if citations else None,
        ),
      )
      session.commit()

  def generate_session_title(
    self,
    client: genai.Client,
    user_text: str,
    model_text: str,
  ) -> str:
    try:
      prompt = (
        f"Summarize Q&A into a short Japanese title (max 20 chars) without 'Title:'.\n"
        f"Q:{user_text[:200]}\nA:{model_text[:200]}"
      )
      resp = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
      return resp.text.strip() if resp.text else "New Chat"
    except Exception as e:  # noqa: BLE001
      logger.warning(f"Title generation failed: {e}")
      return "New Chat"


# ==========================================
# 3. UI Components
# ==========================================


@st.cache_resource
def get_manager():
  return GeminiManager(DB_URL)


def render_sidebar(
  manager: GeminiManager,
) -> tuple[genai.Client | None, str | None, dict, str]:
  with st.sidebar:
    st.title("Gemini Manager")

    page = st.radio(
      "Navigation",
      ["Documents", "RAG Chat", "History Manager"],
      label_visibility="collapsed",
    )
    st.divider()

    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    if not api_key:
      st.warning("API Key required")
      return None, None, {}, page

    client = manager.get_client(api_key)

    # Only show store selector for Documents and Chat
    selected_store_id = None
    store_options = {}
    effective_id = None

    if page in ["Documents", "RAG Chat"]:
      st.subheader("🗄️ Store")
      stores = manager.list_active_stores()
      store_options = {s["store_id"]: f"{s['display_name']} ({s['file_count']} files)" for s in stores}

      selected_store_id = st.selectbox(
        "Select Store",
        options=list(store_options.keys()),
        format_func=lambda x: store_options[x],
        index=0 if stores else None,
      )

      if selected_store_id:
        st.text("Store ID:")
        st.code(selected_store_id, language="text")

      st.divider()

      # Chat Settings (Debug)
      if page == "RAG Chat":
        with st.expander("🛠️ Debug Settings"):
          manual_store_id = st.text_input(
            "Manual Store ID",
            placeholder="fileSearchStores/...",
            key="manual_store_id",
          )
          effective_id = manual_store_id.strip() if manual_store_id else selected_store_id
      else:
        effective_id = selected_store_id

      # Create Store (Available in Documents/Chat)
      with st.expander("➕ Create New Store"):  # noqa: RUF001
        new_store_name = st.text_input("Name", placeholder="my-knowledge-base")
        if st.button("Create"):
          if new_store_name.strip():
            with st.spinner("Creating..."):
              new_id = manager.create_store(client, new_store_name)
            st.success(f"Created {new_id}")
            time.sleep(1)
            st.rerun()
          else:
            st.error("Name required")

  return client, effective_id, store_options, page


def render_documents_page(
  manager: GeminiManager,
  client: genai.Client,
  store_id: str | None,
  store_options: dict,
):
  st.title("📄 Documents Management")
  display_name = store_options.get(store_id, store_id) if store_id else "None"
  st.info(f"Active Store: **{display_name}**")

  if not store_id:
    st.warning("Please create or select a store in the sidebar.")
    return

  col1, col2 = st.columns([2, 1])
  with col1:
    with st.form("upload_form", clear_on_submit=True):
      uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
      )
      submitted = st.form_submit_button("🚀 Upload & Sync")  # noqa: RUF001

    if submitted and uploaded_files:
      progress_bar = st.progress(0, text="Starting upload...")
      results = manager.upload_files(
        client,
        store_id,
        uploaded_files,
        progress_bar,
      )

      if results["uploaded"]:
        st.success(f"✅ Uploaded {len(results['uploaded'])} files.")
      if results["skipped"]:
        st.warning(f"⚠️ Skipped {len(results['skipped'])} duplicates.")
      if results["failed"]:
        st.error(f"❌ Failed {len(results['failed'])} files.")
      time.sleep(2)
      st.rerun()

  with col2:
    st.write("Maintenance")
    if st.button("🔄 Refresh Store"):
      with st.spinner("Refreshing..."):
        new_id = manager.refresh_store(client, store_id)
      st.success(f"Done! New ID: {new_id}")
      time.sleep(1)
      st.rerun()

  st.subheader("Registered Files")
  files = manager.get_store_files(store_id)
  if files:
    df = pd.DataFrame(files)
    st.dataframe(
      df[["file_name", "size", "md5", "created_at"]],
      use_container_width=True,
    )
  else:
    st.write("No files yet.")

  # Danger Zone
  st.divider()
  with st.expander("⚠️ Danger Zone", expanded=False):
    st.warning("Deleting a store is irreversible. Local files will also be deleted.")
    if st.button("🗑️ Delete Store Permanently", type="primary"):
      with st.spinner(f"Deleting {store_id}..."):
        manager.delete_store(client, store_id)
      st.success("Store deleted.")
      time.sleep(1)
      st.rerun()


def render_history_manager_page(manager: GeminiManager):
  st.title("📜 History Manager")
  sessions = manager.list_chat_sessions()

  if not sessions:
    st.info("No chat history found.")
    return

  # Using columns for a simple table layout with delete buttons
  for sess in sessions:
    c1, c2, c3 = st.columns([4, 2, 1])
    with c1:
      st.write(f"**{sess['name']}**")
    with c2:
      st.caption(f"{sess['id']}")
    with c3:
      if st.button("🗑️", key=f"del_{sess['id']}", help="Delete Session"):
        manager.delete_chat_session(sess["id"])
        st.rerun()
    st.divider()


def _render_chat_log(container, messages: list[dict]):
  with container:
    if not messages:
      st.write("Start a conversation...")

    for msg in messages:
      avatar = "🤖" if msg["role"] == "model" else None
      with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("citations"):
          with st.expander("📚 Citations"):
            for c in msg["citations"]:
              st.markdown(f"- {c}")


def _prepare_chat_session(manager: GeminiManager, store_id: str, prompt: str) -> bool:
  """Initialize session if needed. Returns is_new_session."""
  is_new = False
  if st.session_state.active_session_id is None:
    is_new = True
    new_id = manager.create_chat_session("New Chat...", store_id)
    st.session_state.active_session_id = new_id

  current_sess_id = st.session_state.active_session_id
  manager.save_message(str(current_sess_id), "user", prompt)
  return is_new


def _generate_and_stream_response(
  client: genai.Client,
  container,
  store_id: str,
  messages: list[dict],
) -> tuple[str, list[str]]:
  """Generate content stream and render to container."""
  full_text = ""
  captured_citations = []

  with container, st.chat_message("model", avatar="🤖"):
    placeholder = st.empty()
    try:
      tool = types.Tool(
        file_search=types.FileSearch(file_search_store_names=[store_id]),
      )
      config = types.GenerateContentConfig(tools=[tool], temperature=0.3)

      # Cast Content correctly
      contents_list = [types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages[-6:]]

      stream = client.models.generate_content_stream(
        model=DEFAULT_MODEL,
        contents=cast("Any", contents_list),
        config=config,
      )

      for chunk in stream:
        if chunk.text:
          full_text += chunk.text
          placeholder.markdown(full_text + "▌")
        if chunk.candidates and chunk.candidates[0].grounding_metadata:
          gm = chunk.candidates[0].grounding_metadata
          if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
            captured_citations.extend(f"Web: {gc.web.title}" for gc in gm.grounding_chunks if gc.web)

      placeholder.markdown(full_text)
      if captured_citations:
        with st.expander("📚 Citations"):
          for c in captured_citations:
            st.write(c)

    except Exception as e:
      logger.exception("Generation error")
      st.error(f"Error: {e}")

  return full_text, captured_citations


def _finalize_chat_turn(
  manager: GeminiManager,
  client: genai.Client,
  prompt: str,
  full_text: str,
  captured_citations: list[str],
  is_new_session: bool,  # noqa: FBT001
):
  """Save response and update title if needed."""
  if not full_text:
    return

  current_sess_id = st.session_state.active_session_id
  manager.save_message(str(current_sess_id), "model", full_text, captured_citations)

  if is_new_session:
    with st.spinner("Generating title..."):
      title = manager.generate_session_title(client, prompt, full_text)
      manager.update_session_name(str(current_sess_id), title)
    st.rerun()


def _handle_chat_input(
  manager: GeminiManager,
  client: genai.Client,
  store_id: str,
  container,
  messages: list[dict],
):
  # Footer Area for Export Button
  # We place it just above the input widget (visually) by rendering it here.
  # Use columns to push it to the right.
  _, _, _, c_export = st.columns(4)
  with c_export:
    if st.session_state.active_session_id and messages:
      json_data = json.dumps(messages, ensure_ascii=False, indent=2)
      st.download_button(
        label="💾 Export JSON",
        data=json_data,
        file_name=f"chat_{st.session_state.active_session_id}.json",
        mime="application/json",
      )

  prompt = st.chat_input("Ask something... (Enter to send)")
  if not prompt:
    return

  is_new = _prepare_chat_session(manager, store_id, prompt)

  # Update local list for context
  messages.append({"role": "user", "content": prompt})

  with container, st.chat_message("user"):
    st.markdown(prompt)

  full_text, citations = _generate_and_stream_response(
    client,
    container,
    store_id,
    messages,
  )

  _finalize_chat_turn(manager, client, prompt, full_text, citations, is_new)


def render_chat_page(
  manager: GeminiManager,
  client: genai.Client,
  store_id: str | None,
):
  c1, c2 = st.columns([2, 2])
  with c1:
    st.title("💬 RAG Chat")
  with c2:
    st.markdown(f"**Model:** `{DEFAULT_MODEL}`")

  if not store_id:
    st.warning("Please select a store in the sidebar to chat.")
    return

  if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

  sessions = manager.list_chat_sessions()
  session_map = {s["id"]: s["name"] for s in sessions}
  options_keys = [None, *[s["id"] for s in sessions]]  # noqa: RUF005

  try:
    current_index = options_keys.index(st.session_state.active_session_id)
  except ValueError:
    current_index = 0

  def on_session_change():
    st.session_state.active_session_id = st.session_state.history_selector

  st.selectbox(
    "History",
    options=options_keys,
    format_func=lambda x: "✨ New Session" if x is None else session_map.get(x, x),
    index=current_index,
    key="history_selector",
    on_change=on_session_change,
  )

  messages = manager.get_chat_history(st.session_state.active_session_id) if st.session_state.active_session_id else []

  # 500px height for standard laptop compatibility
  chat_container = st.container(height=500, border=True)
  _render_chat_log(chat_container, messages)

  _handle_chat_input(manager, client, store_id, chat_container, messages)


def main():
  st.set_page_config(page_title="Gemini Store Manager v15", layout="wide")
  manager = get_manager()

  client, effective_store_id, store_options, page = render_sidebar(manager)

  if not client:
    return

  if page == "Documents":
    render_documents_page(manager, client, effective_store_id, store_options)
  elif page == "RAG Chat":
    render_chat_page(manager, client, effective_store_id)
  elif page == "History Manager":
    render_history_manager_page(manager)


if __name__ == "__main__":
  main()
