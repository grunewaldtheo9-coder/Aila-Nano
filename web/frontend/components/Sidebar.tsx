"use client";

import { AGENT_ICONS, AGENT_LABELS, AgentInfo } from "@/lib/types";

interface Props {
  agents: AgentInfo[];
  activeAgent: string;
  onSelectAgent: (name: string) => void;
  conversations: string[];
  activeConversation: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({
  agents,
  activeAgent,
  onSelectAgent,
  conversations,
  activeConversation,
  onSelectConversation,
  onNewConversation,
  open,
  onClose,
}: Props) {
  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/30 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed z-30 flex h-full w-72 flex-col border-r border-gray-200 bg-white p-3
          transition-transform dark:border-gray-800 dark:bg-gray-900
          md:static md:translate-x-0
          ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="mb-4 flex items-center gap-2 px-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-aila-500 font-bold text-white">
            A
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">Aila Nano</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Aila Company Solutions</div>
          </div>
        </div>

        <button
          onClick={onNewConversation}
          className="mb-4 flex items-center justify-center gap-2 rounded-lg bg-aila-500 px-3 py-2
            text-sm font-medium text-white hover:bg-aila-600 transition-colors"
        >
          + New conversation
        </button>

        <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-gray-400">
          Agent
        </div>
        <div className="mb-4 flex flex-col gap-1">
          {agents.map((a) => (
            <button
              key={a.name}
              onClick={() => onSelectAgent(a.name)}
              className={`flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors
                ${
                  activeAgent === a.name
                    ? "bg-aila-100 text-aila-800 dark:bg-aila-900/40 dark:text-aila-200"
                    : "hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}
            >
              <span>{AGENT_ICONS[a.name] ?? "🤖"}</span>
              <span>{AGENT_LABELS[a.name] ?? a.name}</span>
            </button>
          ))}
        </div>

        <div className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-gray-400">
          History
        </div>
        <div className="flex-1 overflow-y-auto">
          {conversations.length === 0 && (
            <div className="px-2 py-4 text-xs text-gray-400">No conversations yet.</div>
          )}
          {conversations.map((id) => (
            <button
              key={id}
              onClick={() => onSelectConversation(id)}
              className={`block w-full truncate rounded-lg px-3 py-2 text-left text-sm transition-colors
                ${
                  activeConversation === id
                    ? "bg-gray-100 dark:bg-gray-800"
                    : "hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}
              title={id}
            >
              {id}
            </button>
          ))}
        </div>

        <div className="mt-3 border-t border-gray-200 pt-3 text-xs text-gray-400 dark:border-gray-800">
          ~10.9M params · original architecture
        </div>
      </aside>
    </>
  );
}
