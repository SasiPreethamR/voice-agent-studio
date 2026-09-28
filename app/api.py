"""REST API: documents/RAG, tool library, agents, model provider (Gemini BYOK), health."""

import json
import asyncio
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.config import (
    STT_URL, LLM_URL, TTS_URL, INDIC_URL, GEMINI_LLM_MODEL, GEMINI_STT_MODEL, GEMINI_TTS_MODEL,
    GEMINI_REASONING_EFFORT, GEMINI_API_KEY, MODEL_PROVIDER, GEMINI_ONLY,
)
from app.providers import GEMINI_VOICES, ModelSettings, ProviderError, gemini_list_models
from app.documents import rag
from app.agents import agent_manager
from app.tools import tool_library

router = APIRouter()


def _request_model(request: Request) -> ModelSettings:
    """Model settings the browser attached for background LLM work (K-Quest generation)."""
    return ModelSettings.from_header(request.headers.get("x-model-settings"))


@router.post("/api/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...), relative_path: Optional[str] = Form(None)):
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")
    try:
        doc_id = rag.add_document(file.filename or "document.txt", content, relative_path=relative_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    asyncio.create_task(rag.generate_faqs(doc_id, model=_request_model(request)))
    doc = rag.documents.get(doc_id, {})
    return JSONResponse({"status": "ok", "document": doc})


@router.post("/api/documents/ingest")
async def ingest_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    paths: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    parsed_paths = []
    if paths:
        try:
            parsed_paths = json.loads(paths)
        except Exception:
            raise HTTPException(400, "paths must be a JSON array")
        if not isinstance(parsed_paths, list):
            raise HTTPException(400, "paths must be a JSON array")

    file_items = []
    for index, file in enumerate(files):
        content = await file.read()
        if not content:
            continue
        rel_path = parsed_paths[index] if index < len(parsed_paths) else (file.filename or f"document-{index + 1}")
        for item in rag.expand_archive(file.filename or "archive.zip", content, rel_path):
            file_items.append(item)
    if not file_items:
        raise HTTPException(400, "No non-empty files uploaded")

    collection_id = rag.create_collection(collection_name or "Upload", len(file_items))
    job_id = rag.create_job(collection_id, len(file_items))
    asyncio.create_task(rag.process_ingest_job(job_id, file_items, model=_request_model(request)))
    return JSONResponse({
        "status": "queued",
        "job_id": job_id,
        "collection_id": collection_id,
        "file_count": len(file_items),
    })


@router.get("/api/documents")
async def list_documents():
    return JSONResponse({"documents": rag.list_documents()})


@router.get("/api/documents/tree")
async def documents_tree():
    return JSONResponse({"tree": rag.list_tree(), "collections": rag.list_collections()})


@router.get("/api/documents/jobs")
async def list_document_jobs():
    return JSONResponse({"jobs": rag.list_jobs()})


@router.get("/api/documents/jobs/{job_id}")
async def document_job(job_id: str):
    job = rag.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse({"job": job})


@router.post("/api/documents/collections/{collection_id}/retry")
async def retry_collection_quests(collection_id: str, request: Request):
    """Re-queue K-Quest generation for all ready files whose quest_status is still pending."""
    job_id = rag.retry_collection(collection_id)
    if not job_id:
        return JSONResponse({"status": "nothing_to_retry", "message": "All files already have K-Quest data"})
    asyncio.create_task(rag.run_retry_quests(job_id, collection_id, model=_request_model(request)))
    return JSONResponse({"status": "queued", "job_id": job_id})


@router.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = rag.get_document_view(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return JSONResponse(doc)


@router.delete("/api/documents/collections/{collection_id}")
async def delete_document_collection(collection_id: str):
    ok = rag.delete_collection(collection_id)
    if not ok:
        raise HTTPException(404, "Upload not found")
    return JSONResponse({"status": "deleted"})


@router.delete("/api/documents/folders")
async def delete_document_folder(path: str):
    ok = rag.delete_folder(path)
    if not ok:
        raise HTTPException(404, "Folder not found")
    return JSONResponse({"status": "deleted"})


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    ok = rag.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "Document not found")
    return JSONResponse({"status": "deleted"})


@router.get("/api/rag/search")
async def rag_search(q: str, top_k: int = 5, scopes: Optional[str] = None):
    doc_scopes = None
    if scopes:
        try:
            doc_scopes = json.loads(scopes)
        except Exception:
            raise HTTPException(400, "scopes must be JSON")
    results = rag.retrieve(q, top_k, scopes=doc_scopes)
    return JSONResponse({"results": results})


@router.get("/api/tools")
async def list_tools():
    return JSONResponse({"tools": tool_library.list_all()})


@router.post("/api/tools")
async def add_tool(data: dict):
    if not data.get("name"):
        raise HTTPException(400, "Name is required")
    if not data.get("description"):
        raise HTTPException(400, "Description is required")
    tool = tool_library.add(data)
    return JSONResponse({"status": "ok", "tool": tool})


@router.get("/api/tools/{tool_id}")
async def get_tool(tool_id: str):
    tool = tool_library.get(tool_id)
    if not tool:
        raise HTTPException(404, "Tool not found")
    return JSONResponse(tool)


@router.put("/api/tools/{tool_id}")
async def update_tool(tool_id: str, data: dict):
    tool = tool_library.update(tool_id, data)
    if not tool:
        raise HTTPException(404, "Tool not found")
    return JSONResponse({"status": "ok", "tool": tool})


@router.delete("/api/tools/{tool_id}")
async def delete_tool(tool_id: str):
    if not tool_library.delete(tool_id):
        raise HTTPException(404, "Tool not found")
    return JSONResponse({"status": "deleted"})


@router.get("/api/agents")
async def list_agents():
    return JSONResponse({"agents": agent_manager.list_all()})


@router.post("/api/agents")
async def create_agent(data: dict):
    if not data.get("name"):
        raise HTTPException(400, "Name is required")
    agent = agent_manager.add(data)
    return JSONResponse({"status": "ok", "agent": agent})


@router.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    agent = agent_manager.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return JSONResponse(agent)


@router.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, data: dict):
    agent = agent_manager.update(agent_id, data)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return JSONResponse({"status": "ok", "agent": agent})


@router.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if not agent_manager.delete(agent_id):
        raise HTTPException(404, "Agent not found")
    return JSONResponse({"status": "deleted"})


@router.get("/api/model/defaults")
async def model_defaults():
    return JSONResponse({
        "llm_model": GEMINI_LLM_MODEL,
        "stt_model": GEMINI_STT_MODEL,
        "tts_model": GEMINI_TTS_MODEL,
        "reasoning_effort": GEMINI_REASONING_EFFORT,
        "voices": GEMINI_VOICES,
        "server_key": bool(GEMINI_API_KEY),
        "mode": MODEL_PROVIDER,
    })


@router.post("/api/model/test")
async def model_test(data: dict):
    """Check a Gemini key from this server (also proves outbound access) and list models."""
    try:
        models = await gemini_list_models(ModelSettings.from_dict({**(data or {}), "provider": "gemini"}))
    except ProviderError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except httpx.HTTPError as e:
        return JSONResponse({"ok": False, "error": f"Could not reach the Gemini API from this server: {e}"}, status_code=502)
    return JSONResponse({"ok": True, **models})


@router.get("/health")
async def health():
    if GEMINI_ONLY:
        return {"status": "ok", "mode": MODEL_PROVIDER, "services": {}}
    return {"status": "ok", "mode": MODEL_PROVIDER, "services": {"stt": STT_URL, "llm": LLM_URL, "tts": TTS_URL}}


@router.get("/api/services/status")
async def services_status():
    status = {}
    if GEMINI_ONLY:
        return JSONResponse({"mode": MODEL_PROVIDER, "services": status})
    for name, url in [("stt", STT_URL), ("llm", LLM_URL), ("tts", TTS_URL), ("indic", INDIC_URL)]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{url}/health")
                status[name] = "online" if r.status_code == 200 else "error"
        except Exception:
            status[name] = "offline"
    return JSONResponse({"mode": MODEL_PROVIDER, "services": status})
