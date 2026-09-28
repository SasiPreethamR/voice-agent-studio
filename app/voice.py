"""Real-time voice calls over /ws/voice.

Energy-based VAD and barge-in, then per turn: STT -> RAG -> LLM (with tools)
-> sentence-by-sentence TTS, streamed back to the browser in order.
"""

import re
import json
import uuid
import base64
import asyncio
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import INPUT_SAMPLE_RATE, STATIC_DIR
from app.providers import (
    LANG_NAMES, KOKORO_LANG_CODES, ModelSettings, ProviderError, transcribe_audio,
    stream_llm_tokens, call_llm_with_tools, synthesize_speech_stream,
)
from app.documents import rag
from app.agents import agent_manager
from app.tools import tool_library, execute_tool_call, _parse_raw_tool_call
from app.handoff import (
    get_handoff_openai_tool, _tool_list_has_handoff, handoff_manager, ACTIVE_VOICE_SESSIONS,
    _send_audio_to_operators,
)

router = APIRouter()


# -- VAD constants --
ENERGY_THRESHOLD = 600


INTERRUPT_ENERGY_THRESHOLD = 700


SILENCE_TURN_END_MS = 800


CHUNK_MS = 100


MIN_SPEECH_FRAMES = 3


MIN_INTERRUPT_FRAMES = 3


SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. You are having a live voice "
    "conversation. Keep your answers concise (1-3 sentences unless more detail "
    "is requested). Use natural, conversational language. Avoid markdown, lists, "
    "or formatting. If context from documents is provided, use it as the primary "
    "source for document-specific questions. If the retrieved document context is "
    "missing or insufficient, say that you do not see it in the selected documents "
    "instead of guessing."
)


# Pre-load tool-wait audio cue (PCM, 24kHz)
TOOL_WAIT_AUDIO_B64 = ""


_twpath = STATIC_DIR / "tool_wait.pcm"


if _twpath.exists():
    TOOL_WAIT_AUDIO_B64 = base64.b64encode(_twpath.read_bytes()).decode()
    print(f"[ORCH] Loaded tool-wait audio ({len(TOOL_WAIT_AUDIO_B64)//1024}KB b64)")


class VoiceSession:
    def __init__(self):
        self.session_id = "sess_" + uuid.uuid4().hex[:12]
        self.active = True
        self.state = "idle"
        self.audio_buffer = bytearray()
        self.pre_buffer = bytearray()
        self.is_user_speaking = False
        self.speech_frame_count = 0
        self.interrupt_frame_count = 0
        self.silence_frames = 0
        self.chat_history: list = []
        self.active_agent_id: Optional[str] = None
        self.language = "en"
        self.voice = "alloy"
        self.speed = 1.0
        self.interrupt_event = asyncio.Event()
        self.turn_complete_event = asyncio.Event()
        self.partial_text = ""
        self._llm_task: Optional[asyncio.Task] = None
        self.pending_tts_tasks: list = []
        self._ws: Optional[WebSocket] = None
        self.active_handoff_id: Optional[str] = None
        self.handoff_connected = False
        self.model = ModelSettings()

    def apply_agent(self, agent_id: str):
        """Load agent settings into this session."""
        ag = agent_manager.get(agent_id)
        if ag:
            self.active_agent_id = agent_id
            self.language = ag.get("language", "en")
            self.voice = ag.get("voice", "alloy")
            self.speed = ag.get("speed", 1.0)

    def get_agent_instructions(self) -> str:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("instructions", "")
        return ""

    def get_agent_tool_ids(self) -> list:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("tool_ids", [])
        return []

    def get_agent_document_scopes(self) -> list:
        if self.active_agent_id:
            ag = agent_manager.get(self.active_agent_id)
            if ag:
                return ag.get("document_scopes", [])
        return []


def calc_energy(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def split_first_sentence(buf: str, is_first: bool = False):
    min_len = 8 if is_first else 15
    # First sentence end at or past min_len, so a short opener ("Sure.") joins
    # the next sentence instead of holding the whole reply back.
    for m in re.finditer(r'[.!?\u0964](?:\s|$)', buf):
        if m.end() >= min_len:
            sentence = buf[: m.end()].strip()
            remainder = buf[m.end() :].lstrip()
            return sentence, remainder
    if is_first and len(buf) > 40:
        m2 = re.search(r'([^,;]*[,;])\s', buf)
        if m2 and m2.end() >= 20:
            sentence = buf[: m2.end()].strip()
            remainder = buf[m2.end() :].lstrip()
            return sentence, remainder
    return None, buf


@router.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    await ws.accept()
    session = VoiceSession()
    session._ws = ws
    ACTIVE_VOICE_SESSIONS[session.session_id] = session

    partial_task = asyncio.create_task(_partial_stt_worker(ws, session))
    response_task = asyncio.create_task(_response_worker(ws, session))

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]
                if session.state in ("listening", "speaking", "processing"):
                    await _process_audio_chunk(chunk, session, ws)
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                action = msg.get("action")

                if action == "start_call":
                    session.state = "listening"
                    session.chat_history = []
                    session.audio_buffer.clear()
                    session.active_handoff_id = None
                    session.handoff_connected = False
                    session.model = ModelSettings.from_dict(msg.get("model"))
                    print(f"[ORCH] Call started with model={session.model.describe()}")
                    agent_id = msg.get("agent_id")
                    if agent_id:
                        session.apply_agent(agent_id)
                    await ws.send_json({"type": "status", "state": "listening"})

                elif action == "set_model":
                    session.model = ModelSettings.from_dict(msg.get("model"))
                    print(f"[ORCH] Model switched to {session.model.describe()}")
                    await ws.send_json({"type": "model_set", "provider": session.model.provider})

                elif action == "end_call":
                    session.state = "idle"
                    session.interrupt_event.set()
                    session.audio_buffer.clear()
                    session.active_handoff_id = None
                    session.handoff_connected = False
                    await ws.send_json({"type": "status", "state": "idle"})

                elif action == "switch_agent":
                    agent_id = msg.get("agent_id")
                    if agent_id:
                        session.apply_agent(agent_id)
                        await ws.send_json({"type": "agent_switched", "agent_id": agent_id})

                elif action == "handoff_audio":
                    handoff_id = msg.get("handoff_id") or session.active_handoff_id
                    handoff = handoff_manager.get(handoff_id) if handoff_id else None
                    if handoff and handoff.get("status") == "accepted" and handoff_id == session.active_handoff_id:
                        await _send_audio_to_operators(handoff_id, msg.get("audio", ""), int(msg.get("sample_rate") or INPUT_SAMPLE_RATE))

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect message" not in str(e):
            raise
    finally:
        session.active = False
        ACTIVE_VOICE_SESSIONS.pop(session.session_id, None)
        partial_task.cancel()
        response_task.cancel()


async def _process_audio_chunk(chunk: bytes, session: VoiceSession, ws: WebSocket):
    energy = calc_energy(chunk)

    if session.state == "speaking":
        if energy > INTERRUPT_ENERGY_THRESHOLD:
            session.interrupt_frame_count += 1
        else:
            session.interrupt_frame_count = 0

        if session.interrupt_frame_count >= MIN_INTERRUPT_FRAMES:
            print(f"[ORCH] INTERRUPT - strong user speech ({session.interrupt_frame_count} frames, energy={energy:.0f})")
            session.interrupt_event.set()
            for t in session.pending_tts_tasks:
                t.cancel()
            session.pending_tts_tasks.clear()
            session.audio_buffer.clear()
            session.pre_buffer.clear()
            session.is_user_speaking = False
            session.speech_frame_count = 0
            session.interrupt_frame_count = 0
            try:
                await ws.send_json({"type": "interrupt"})
            except Exception:
                pass
            session.state = "listening"
            await ws.send_json({"type": "status", "state": "listening"})
        return

    if energy > ENERGY_THRESHOLD:
        session.interrupt_frame_count = 0
        session.silence_frames = 0
        session.speech_frame_count += 1
        if not session.is_user_speaking:
            session.is_user_speaking = True
            if session.pre_buffer:
                session.audio_buffer.extend(session.pre_buffer)
                session.pre_buffer.clear()
        session.audio_buffer.extend(chunk)
    else:
        session.interrupt_frame_count = 0
        if session.is_user_speaking:
            session.audio_buffer.extend(chunk)
            session.silence_frames += 1
            frames_for_turn_end = max(1, SILENCE_TURN_END_MS // CHUNK_MS)
            if session.silence_frames >= frames_for_turn_end:
                if session.speech_frame_count >= MIN_SPEECH_FRAMES:
                    session.turn_complete_event.set()
                else:
                    print(f"[ORCH] Discarded noise ({session.speech_frame_count} frames)")
                    session.audio_buffer.clear()
                session.is_user_speaking = False
                session.speech_frame_count = 0
        else:
            max_pre = INPUT_SAMPLE_RATE * 2 // 5
            session.pre_buffer.extend(chunk)
            if len(session.pre_buffer) > max_pre:
                session.pre_buffer = session.pre_buffer[-max_pre:]


async def _partial_stt_worker(ws: WebSocket, session: VoiceSession):
    while session.active:
        await asyncio.sleep(1.5)
        if session.model.gemini("stt") and not session.model.live_captions:
            continue  # each partial is a paid API call; opt-in via Model settings
        if (session.state == "listening" and session.is_user_speaking
                and len(session.audio_buffer) > INPUT_SAMPLE_RATE * 2):
            try:
                text = await transcribe_audio(bytes(session.audio_buffer), session.language, model=session.model)
                if text:
                    session.partial_text = text
                    await ws.send_json({"type": "stt_partial", "text": text})
            except Exception:
                pass


def _build_system_msg(session: VoiceSession, rag_context: str) -> str:
    system_msg = SYSTEM_PROMPT
    agent_instructions = session.get_agent_instructions()
    if agent_instructions:
        system_msg += "\n\n[Agent Instructions - follow these strictly]\n" + agent_instructions
    if session.language != "en":
        lang_name = LANG_NAMES.get(session.language, session.language)
        system_msg += f"\nIMPORTANT: You MUST respond in {lang_name}. All your replies should be in {lang_name}."
    if rag_context:
        system_msg += "\n\n" + rag_context
    return system_msg


async def _run_tool_calls(ws: WebSocket, session: VoiceSession, tool_calls: list) -> tuple:
    """Execute model tool calls and append their results to history.

    Returns (switched_agent_id, handoff_requested).
    """
    switched_agent_id = None
    handoff_requested = False

    # Tell frontend to pause mic and play wait audio
    await ws.send_json({"type": "tool_executing", "audio": TOOL_WAIT_AUDIO_B64 if TOOL_WAIT_AUDIO_B64 else None})

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        raw_args = tc["function"].get("arguments") or "{}"
        fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        print(f"[ORCH] Tool call: {fn_name}({fn_args})")
        await ws.send_json({"type": "tool_call", "name": fn_name, "args": fn_args})

        result = await execute_tool_call(fn_name, fn_args, session=session)
        print(f"[ORCH] Tool result: {result[:200]}")
        await ws.send_json({"type": "tool_result", "name": fn_name, "result": result[:2000]})

        try:
            if json.loads(result).get("status") == "handoff_requested":
                handoff_requested = True
        except Exception:
            pass

        # Check if it was a switch_agent call
        if fn_name == "switch_agent":
            try:
                rj = json.loads(result)
                if rj.get("status") == "switched":
                    switched_agent_id = rj["agent_id"]
            except Exception:
                pass

        session.chat_history.append({
            "role": "tool", "tool_call_id": tc["id"], "content": result,
        })

    # Notify frontend about agent switch
    if switched_agent_id:
        await ws.send_json({"type": "agent_switched", "agent_id": switched_agent_id})
    return switched_agent_id, handoff_requested


def _merge_tool_call_deltas(acc: dict, deltas: list):
    """Accumulate streamed OpenAI-style tool_call deltas (keyed by index)."""
    for d in deltas or []:
        idx = d.get("index")
        if idx is None:
            idx = len(acc)
        cur = acc.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if d.get("id"):
            cur["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name") and not cur["function"]["name"]:
            cur["function"]["name"] = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, dict):
            cur["function"]["arguments"] = json.dumps(args)
        elif args:
            cur["function"]["arguments"] += args
        if d.get("extra_content"):
            cur["extra_content"] = d["extra_content"]  # Gemini thought signature; must be sent back


def _finalize_tool_calls(acc: dict) -> list:
    calls = []
    for idx in sorted(acc):
        tc = acc[idx]
        if not tc["function"]["name"]:
            continue
        tc["id"] = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
        tc["function"]["arguments"] = tc["function"]["arguments"] or "{}"
        calls.append(tc)
    return calls


async def _response_worker(ws: WebSocket, session: VoiceSession):
    while session.active:
        await session.turn_complete_event.wait()
        session.turn_complete_event.clear()
        if session.state not in ("listening", "speaking"):
            continue
        audio_data = bytes(session.audio_buffer)
        session.audio_buffer.clear()
        session.partial_text = ""
        if len(audio_data) < INPUT_SAMPLE_RATE:
            continue

        # 1. Transcribe
        session.state = "processing"
        await ws.send_json({"type": "status", "state": "processing"})
        t_pipeline = time.time()
        try:
            t0 = time.time()
            user_text = await transcribe_audio(audio_data, session.language, model=session.model)
            print(f"[ORCH] STT: {time.time()-t0:.2f}s -> '{user_text[:60]}'")
        except Exception as e:
            user_text = ""
            print(f"[ORCH] STT error: {e}")
            if isinstance(e, ProviderError):
                await ws.send_json({"type": "error", "message": f"Speech-to-text: {e}"})
        if not user_text:
            session.state = "listening"
            await ws.send_json({"type": "status", "state": "listening"})
            continue
        await ws.send_json({"type": "stt_final", "text": user_text})
        session.chat_history.append({"role": "user", "content": user_text})

        # 2. RAG
        t0 = time.time()
        rag_context = ""
        try:
            doc_scopes = session.get_agent_document_scopes()
            results = rag.retrieve(user_text, top_k=3, scopes=doc_scopes if doc_scopes else None)
            if results:
                rag_context = "\n\n[Relevant context from documents]\n" + "\n---\n".join(r["text"] for r in results)
        except Exception as e:
            print(f"[ORCH] RAG error: {e}")
        print(f"[ORCH] RAG: {time.time()-t0:.3f}s")

        # 3. Build messages with agent context
        system_msg = _build_system_msg(session, rag_context)

        # Get tools filtered to this agent's tool_ids
        agent_tool_ids = session.get_agent_tool_ids()
        tools = tool_library.to_openai_tools(agent_tool_ids if agent_tool_ids else None) or []
        if not _tool_list_has_handoff(tools):
            tools.append(get_handoff_openai_tool())
        # Auto-inject switch_agent tool if multiple agents
        if len(agent_manager.agents) > 1:
            tools.append(agent_manager.get_switch_tool())
        tools = tools or None

        if tools:
            system_msg += ("\n\nYou have access to tools/APIs. Use them when the user's request matches a tool's purpose. "
                          "After receiving tool results, give a natural conversational summary - do NOT repeat raw data.")

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(session.chat_history[-20:])

        # 4. LLM - check for tool calls. Gemini skips this round trip: the first
        # streamed call carries the tools, so speech starts with the first sentence.
        tool_call_response = None
        stream_tools = None  # Gemini needs the tool declarations whenever history holds tool calls
        gemini_stream_tools = tools if session.model.gemini("llm") else None
        if tools and not gemini_stream_tools:
            try:
                t0 = time.time()
                llm_resp = await call_llm_with_tools(messages, tools, model=session.model)
                choice = llm_resp["choices"][0]
                msg_obj = choice["message"]
                finish_reason = choice.get("finish_reason", "")
                print(f"[ORCH] LLM (tool check): {time.time()-t0:.2f}s, finish={finish_reason}")

                if msg_obj.get("tool_calls"):
                    session.chat_history.append(msg_obj)
                    switched_agent_id, handoff_requested = await _run_tool_calls(ws, session, msg_obj["tool_calls"])

                    if handoff_requested:
                        session.state = "handoff_waiting"
                        continue

                    # Rebuild messages for NL reply
                    if switched_agent_id:
                        system_msg = _build_system_msg(session, rag_context)

                    messages = [{"role": "system", "content": system_msg}]
                    messages.extend(session.chat_history[-20:])
                    if session.model.gemini("llm"):
                        stream_tools = tools
                    tools = None

                    # Tell frontend tool execution done, resume mic
                    await ws.send_json({"type": "tool_done"})

                elif msg_obj.get("content"):
                    raw_content = msg_obj["content"]
                    raw_fn, raw_args = _parse_raw_tool_call(raw_content)
                    if raw_fn and raw_args:
                        print(f"[ORCH] Fallback raw tool call: {raw_fn}({raw_args})")
                        name_map = tool_library.get_name_mapping()
                        display_name = name_map.get(raw_fn, raw_fn)
                        handoff_requested = False
                        await ws.send_json({"type": "tool_executing", "audio": TOOL_WAIT_AUDIO_B64 if TOOL_WAIT_AUDIO_B64 else None})
                        await ws.send_json({"type": "tool_call", "name": display_name, "args": raw_args})
                        result = await execute_tool_call(raw_fn, raw_args, session=session)
                        print(f"[ORCH] Tool result: {result[:200]}")
                        await ws.send_json({"type": "tool_result", "name": display_name, "result": result[:2000]})
                        try:
                            if json.loads(result).get("status") == "handoff_requested":
                                handoff_requested = True
                        except Exception:
                            pass

                        if raw_fn == "switch_agent":
                            try:
                                rj = json.loads(result)
                                if rj.get("status") == "switched":
                                    await ws.send_json({"type": "agent_switched", "agent_id": rj["agent_id"]})
                            except Exception:
                                pass

                        session.chat_history.append({"role": "assistant", "content": raw_content})
                        session.chat_history.append({"role": "user", "content": f"[Tool '{display_name}' returned]: {result}\n\nNow give a brief natural language summary of the result to the user. Do NOT output any tool calls."})
                        messages = [{"role": "system", "content": system_msg}]
                        messages.extend(session.chat_history[-20:])
                        tools = None
                        if handoff_requested:
                            session.state = "handoff_waiting"
                            continue
                        await ws.send_json({"type": "tool_done"})
                    else:
                        tool_call_response = raw_content
            except Exception as e:
                print(f"[ORCH] Tool-check LLM error: {e}")
                if isinstance(e, ProviderError):
                    await ws.send_json({"type": "error", "message": f"LLM: {e}"})
                    session.state = "listening"
                    await ws.send_json({"type": "status", "state": "listening"})
                    continue

        # 5. Stream LLM -> sentence -> TTS
        session.state = "speaking"
        session.interrupt_event.clear()
        session.pending_tts_tasks = []
        await ws.send_json({"type": "status", "state": "speaking"})

        full_response = ""
        sentence_buf = ""
        is_first_sentence = True
        t_llm_start = time.time()
        first_token_time = None
        tts_lang_code = KOKORO_LANG_CODES.get(session.language, "a")

        tts_condition = asyncio.Condition()
        # seq -> {"chunks": [pcm, ...], "done": bool, "text": str, "t0": float, "first": float|None}
        tts_entries: dict[int, dict] = {}
        tts_state = {"issued": 0, "closed": False, "error_sent": False, "first_audio": False}

        async def _tts_and_store(sequence: int, text: str, voice: str, speed: float):
            entry = tts_entries[sequence]
            try:
                if session.interrupt_event.is_set():
                    return
                async for chunk in synthesize_speech_stream(
                    text, voice, speed,
                    lang_code=tts_lang_code,
                    language=session.language,
                    model=session.model,
                ):
                    if session.interrupt_event.is_set():
                        return
                    async with tts_condition:
                        if entry["first"] is None:
                            entry["first"] = time.time()
                        entry["chunks"].append(chunk)
                        tts_condition.notify_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ORCH] TTS failed for seq={sequence}: {exc}")
                if isinstance(exc, ProviderError) and not tts_state["error_sent"]:
                    tts_state["error_sent"] = True
                    try:
                        await ws.send_json({"type": "error", "message": f"Text-to-speech: {exc}"})
                    except Exception:
                        pass
            finally:
                async with tts_condition:
                    entry["done"] = True
                    tts_condition.notify_all()

        def _queue_tts(text: str):
            text = (text or "").strip()
            if not text:
                return
            sequence = tts_state["issued"]
            tts_state["issued"] += 1
            tts_entries[sequence] = {"chunks": [], "done": False, "text": text, "t0": time.time(), "first": None}
            task = asyncio.create_task(_tts_and_store(sequence, text, session.voice, session.speed))
            session.pending_tts_tasks.append(task)

        async def _tts_ordered_sender():
            # Sentences synthesize in parallel; audio is forwarded strictly in
            # order, chunk by chunk, as soon as the current sentence produces it.
            next_sequence = 0
            while True:
                async with tts_condition:
                    await tts_condition.wait_for(
                        lambda: session.interrupt_event.is_set()
                        or (next_sequence in tts_entries
                            and (tts_entries[next_sequence]["chunks"] or tts_entries[next_sequence]["done"]))
                        or (tts_state["closed"] and next_sequence >= tts_state["issued"])
                    )
                    if session.interrupt_event.is_set():
                        return
                    if next_sequence not in tts_entries:
                        return  # closed and everything sent
                    entry = tts_entries[next_sequence]
                    chunks, entry["chunks"] = entry["chunks"], []
                    finished = entry["done"] and not entry["chunks"]
                    current_sequence = next_sequence
                for chunk in chunks:
                    if session.interrupt_event.is_set():
                        return
                    await ws.send_json({"type": "tts_audio", "audio": base64.b64encode(chunk).decode(), "seq": current_sequence})
                    if not tts_state["first_audio"]:
                        tts_state["first_audio"] = True
                        print(f"[ORCH] First audio: {time.time() - t_pipeline:.2f}s after end of speech "
                              f"(+{SILENCE_TURN_END_MS / 1000:.1f}s silence detection)")
                if finished:
                    if entry["first"] is not None:
                        print(f"[ORCH] TTS seq={current_sequence} '{entry['text'][:40]}...' -> "
                              f"first chunk {entry['first'] - entry['t0']:.2f}s, done {time.time() - entry['t0']:.2f}s")
                    tts_entries.pop(current_sequence, None)
                    next_sequence += 1

        session.pending_tts_tasks.append(asyncio.create_task(_tts_ordered_sender()))

        async def _stream_pass(pass_messages: list, pass_tools: Optional[list], tool_choice: Optional[str]) -> tuple:
            """Stream one LLM completion into the UI + TTS. Returns (interrupted, tool_calls)."""
            nonlocal full_response, sentence_buf, is_first_sentence, first_token_time
            tool_acc: dict = {}
            thinking_buffer = ""
            in_thinking = True
            async for token_or_tc in stream_llm_tokens(pass_messages, pass_tools, model=session.model, tool_choice=tool_choice):
                if isinstance(token_or_tc, dict) and "tool_calls" in token_or_tc:
                    _merge_tool_call_deltas(tool_acc, token_or_tc["tool_calls"])
                    continue
                token = token_or_tc
                if in_thinking:
                    thinking_buffer += token
                    stripped = thinking_buffer.lstrip()
                    if stripped.startswith("<start_of_thought>"):
                        if "<end_of_thought>" in stripped:
                            after = stripped.split("<end_of_thought>", 1)[1].lstrip()
                            thinking_buffer = ""
                            in_thinking = False
                            if after:
                                token = after
                            else:
                                continue
                        else:
                            continue
                    elif stripped.lower().startswith("thought"):
                        rest = stripped[7:].lstrip()
                        thinking_buffer = ""
                        in_thinking = False
                        if rest:
                            token = rest
                        else:
                            continue
                    elif len(stripped) > 10:
                        token = thinking_buffer
                        thinking_buffer = ""
                        in_thinking = False
                    else:
                        continue

                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[ORCH] LLM TTFT: {first_token_time - t_llm_start:.3f}s")

                if session.interrupt_event.is_set():
                    print("[ORCH] LLM stream interrupted by user")
                    return True, []

                full_response += token
                sentence_buf += token
                await ws.send_json({"type": "llm_token", "token": token})
                sentence, sentence_buf = split_first_sentence(sentence_buf, is_first_sentence)
                if sentence:
                    is_first_sentence = False
                    _queue_tts(sentence)

            if in_thinking and thinking_buffer.strip() and not thinking_buffer.lstrip().startswith("<start_of_thought>"):
                # Short reply that never passed the thought-prefix check.
                full_response += thinking_buffer
                sentence_buf += thinking_buffer
                await ws.send_json({"type": "llm_token", "token": thinking_buffer})
            if sentence_buf.strip() and not session.interrupt_event.is_set():
                await ws.send_json({"type": "llm_token", "token": ""})
                _queue_tts(sentence_buf.strip())
                sentence_buf = ""
            return False, _finalize_tool_calls(tool_acc)

        interrupted = False
        history_saved = False
        try:
            if tool_call_response:
                full_response = tool_call_response
                stripped = full_response.lstrip()
                if stripped.lower().startswith("thought"):
                    full_response = stripped[7:].lstrip()
                if "<start_of_thought>" in full_response and "<end_of_thought>" in full_response:
                    full_response = full_response.split("<end_of_thought>", 1)[1].lstrip()
                await ws.send_json({"type": "llm_token", "token": full_response})
                remaining = full_response
                while remaining:
                    sent, remaining = split_first_sentence(remaining, is_first_sentence)
                    if sent:
                        is_first_sentence = False
                        _queue_tts(sent)
                    else:
                        if remaining.strip():
                            _queue_tts(remaining.strip())
                        break
            elif gemini_stream_tools:
                # Gemini: one streamed call with tools. Text is spoken as it
                # arrives; if the model calls tools, run them and stream the reply.
                t0 = time.time()
                interrupted, tool_calls = await _stream_pass(messages, gemini_stream_tools, None)
                print(f"[ORCH] LLM stream: {time.time()-t0:.2f}s, tool_calls={len(tool_calls)}")
                if tool_calls and not interrupted:
                    session.chat_history.append({"role": "assistant", "content": full_response or None, "tool_calls": tool_calls})
                    history_saved = True
                    switched_agent_id, handoff_requested = await _run_tool_calls(ws, session, tool_calls)
                    if handoff_requested:
                        session.state = "handoff_waiting"
                    else:
                        if switched_agent_id:
                            system_msg = _build_system_msg(session, rag_context)
                        messages = [{"role": "system", "content": system_msg}]
                        messages.extend(session.chat_history[-20:])
                        await ws.send_json({"type": "tool_done"})
                        reply_start = len(full_response)
                        interrupted, _ = await _stream_pass(messages, gemini_stream_tools, "none")
                        if full_response[reply_start:].strip():
                            session.chat_history.append({"role": "assistant", "content": full_response[reply_start:]})
            else:
                interrupted, _ = await _stream_pass(messages, stream_tools, "none")

            async with tts_condition:
                tts_state["closed"] = True
                tts_condition.notify_all()

            if session.pending_tts_tasks and interrupted:
                for task in session.pending_tts_tasks:
                    task.cancel()
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)
            elif session.pending_tts_tasks:
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)

            int_str = "(interrupted)" if interrupted else ""
            print(f"[ORCH] Pipeline: {time.time()-t_llm_start:.2f}s {int_str}")
        except Exception as e:
            async with tts_condition:
                tts_state["closed"] = True
                tts_condition.notify_all()
            for task in session.pending_tts_tasks:
                if not task.done():
                    task.cancel()
            if session.pending_tts_tasks:
                await asyncio.gather(*session.pending_tts_tasks, return_exceptions=True)
            print(f"[ORCH] LLM/TTS pipeline error: {e}")
            await ws.send_json({"type": "error", "message": str(e)})
        finally:
            session.pending_tts_tasks = []

        if full_response and not history_saved:
            session.chat_history.append({"role": "assistant", "content": full_response})

        await ws.send_json({"type": "llm_done"})
        if session.state == "speaking":
            session.state = "listening"
            session.interrupt_event.clear()
            await ws.send_json({"type": "status", "state": "listening"})
