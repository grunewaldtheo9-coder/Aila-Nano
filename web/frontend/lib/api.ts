import type { AgentInfo, ChatMessage, GenerationSettings, HealthInfo } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthInfo> {
  return jsonFetch("/health");
}

export function listAgents(): Promise<AgentInfo[]> {
  return jsonFetch("/agents");
}

export function getConversationHistory(
  conversationId: string
): Promise<{ conversation_id: string; messages: ChatMessage[] }> {
  return jsonFetch(`/memory/conversations/${encodeURIComponent(conversationId)}`);
}

export function listConversations(): Promise<string[]> {
  return jsonFetch("/memory/conversations");
}

export function clearConversation(conversationId: string): Promise<void> {
  return jsonFetch(`/memory/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
}

export function chatOnce(
  conversationId: string,
  message: string,
  agent: string,
  settings: GenerationSettings
): Promise<{ reply: string }> {
  return jsonFetch("/chat", {
    method: "POST",
    body: JSON.stringify({ conversation_id: conversationId, message, agent, settings }),
  });
}

export async function uploadFile(file: File): Promise<{ filename: string; chunks_indexed: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Streams a chat reply via the backend's SSE endpoint. Browsers'
 * built-in EventSource can't send a POST body, so we parse the
 * text/event-stream response manually from a fetch ReadableStream.
 */
export async function streamChat(
  conversationId: string,
  message: string,
  agent: string,
  settings: GenerationSettings,
  onDelta: (delta: string) => void,
  onDone: () => void,
  onError: (err: string) => void
): Promise<void> {
  try {
    const res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId, message, agent, settings }),
    });
    if (!res.ok || !res.body) {
      onError(await res.text().catch(() => "Stream request failed"));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const raw of events) {
        const lines = raw.split("\n");
        let event = "message";
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        if (event === "token") {
          const parsed = JSON.parse(data);
          onDelta(parsed.delta);
        } else if (event === "done") {
          onDone();
          return;
        } else if (event === "error") {
          const parsed = JSON.parse(data);
          onError(parsed.error);
          return;
        }
      }
    }
    onDone();
  } catch (e) {
    onError(e instanceof Error ? e.message : String(e));
  }
}
