"use client";

import { useEffect, useRef } from "react";
import { AGENT_LABELS, ChatMessage, HealthInfo } from "@/lib/types";
import MessageBubble from "./MessageBubble";
import Composer from "./Composer";
import ThemeToggle from "./ThemeToggle";

interface Props {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onUpload: (file: File) => void;
  agent: string;
  conversationId: string;
  health: HealthInfo | null;
  busy: boolean;
  onOpenSidebar: () => void;
  onOpenSettings: () => void;
}

export default function ChatWindow({
  messages,
  onSend,
  onUpload,
  agent,
  conversationId,
  health,
  busy,
  onOpenSidebar,
  onOpenSettings,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-full flex-1 flex-col">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-3 dark:border-gray-800 dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenSidebar}
            className="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 md:hidden"
            aria-label="Open sidebar"
          >
            ☰
          </button>
          <div>
            <div className="text-sm font-semibold">{AGENT_LABELS[agent] ?? agent}</div>
            <div className="text-xs text-gray-400">{conversationId}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {health && (
            <span
              className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-xs sm:flex
                ${
                  health.model_loaded
                    ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"
                    : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
                }`}
              title={health.model_loaded ? "Serving a trained checkpoint" : "Serving an untrained model — train Aila Nano for real responses"}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {health.model_loaded ? "trained" : "untrained"} · {health.device}
            </span>
          )}
          <button
            onClick={onOpenSettings}
            className="rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800"
            aria-label="Settings"
            title="Generation settings"
          >
            ⚙️
          </button>
          <ThemeToggle />
        </div>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto px-3 py-4 sm:px-6">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center text-gray-400">
            <div className="mb-3 text-4xl">🤖</div>
            <div className="text-sm">Say hello to Aila Nano.</div>
            <div className="mt-1 text-xs">Built from scratch by Aila Company Solutions.</div>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <Composer onSend={onSend} onUpload={onUpload} disabled={busy} />
    </div>
  );
}
