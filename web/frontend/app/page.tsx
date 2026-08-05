"use client";

import { useCallback, useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow from "@/components/ChatWindow";
import SettingsPanel from "@/components/SettingsPanel";
import {
  chatOnce,
  getConversationHistory,
  getHealth,
  listAgents,
  listConversations,
  streamChat,
  uploadFile,
} from "@/lib/api";
import { AgentInfo, ChatMessage, DEFAULT_SETTINGS, GenerationSettings, HealthInfo } from "@/lib/types";

function newConversationId(): string {
  return `conv-${Math.random().toString(36).slice(2, 10)}`;
}

export default function Home() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [agent, setAgent] = useState("general");
  const [conversations, setConversations] = useState<string[]>([]);
  const [conversationId, setConversationId] = useState(newConversationId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [streaming, setStreaming] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setNotice("Could not reach the Aila Nano backend. Is it running?"));
    listAgents()
      .then(setAgents)
      .catch(() => {});
    listConversations()
      .then(setConversations)
      .catch(() => {});
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    setConversationId(id);
    setSidebarOpen(false);
    try {
      const { messages } = await getConversationHistory(id);
      setMessages(messages);
    } catch {
      setMessages([]);
    }
  }, []);

  const handleNewConversation = () => {
    const id = newConversationId();
    setConversationId(id);
    setMessages([]);
    setSidebarOpen(false);
  };

  const refreshConversations = useCallback(() => {
    listConversations()
      .then(setConversations)
      .catch(() => {});
  }, []);

  const handleSend = async (text: string) => {
    setNotice(null);
    const userMsg: ChatMessage = { role: "user", content: text };
    const pendingMsg: ChatMessage = { role: "assistant", content: "", agent_type: agent, pending: true };
    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setBusy(true);

    if (streaming) {
      let acc = "";
      await streamChat(
        conversationId,
        text,
        agent,
        settings,
        (delta) => {
          acc += delta;
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { role: "assistant", content: acc, agent_type: agent };
            return next;
          });
        },
        () => {
          setBusy(false);
          refreshConversations();
        },
        (err) => {
          setNotice(`Generation failed: ${err}`);
          setBusy(false);
        }
      );
    } else {
      try {
        const { reply } = await chatOnce(conversationId, text, agent, settings);
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = { role: "assistant", content: reply, agent_type: agent };
          return next;
        });
        refreshConversations();
      } catch (e) {
        setNotice(`Generation failed: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setBusy(false);
      }
    }
  };

  const handleUpload = async (file: File) => {
    try {
      const res = await uploadFile(file);
      setNotice(`Indexed "${res.filename}" into Aila Nano's knowledge base (${res.chunks_indexed} chunk(s)).`);
    } catch (e) {
      setNotice(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div className="flex h-full">
      <Sidebar
        agents={agents}
        activeAgent={agent}
        onSelectAgent={setAgent}
        conversations={conversations}
        activeConversation={conversationId}
        onSelectConversation={loadConversation}
        onNewConversation={handleNewConversation}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        {notice && (
          <div className="flex items-center justify-between bg-amber-100 px-4 py-2 text-xs text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} className="ml-3 font-bold">
              ✕
            </button>
          </div>
        )}
        <ChatWindow
          messages={messages}
          onSend={handleSend}
          onUpload={handleUpload}
          agent={agent}
          conversationId={conversationId}
          health={health}
          busy={busy}
          onOpenSidebar={() => setSidebarOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </div>

      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        streaming={streaming}
        onStreamingChange={setStreaming}
      />
    </div>
  );
}
