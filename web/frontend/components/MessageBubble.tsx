import { AGENT_ICONS, ChatMessage } from "@/lib/types";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[85%] gap-2 sm:max-w-[70%] ${isUser ? "flex-row-reverse" : ""}`}>
        <div
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-sm
            ${isUser ? "bg-aila-500 text-white" : "bg-gray-200 dark:bg-gray-800"}`}
        >
          {isUser ? "🧑" : AGENT_ICONS[message.agent_type ?? "general"] ?? "🤖"}
        </div>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed prose-chat
            ${
              isUser
                ? "bg-aila-500 text-white rounded-tr-sm"
                : "bg-white text-gray-900 dark:bg-gray-800 dark:text-gray-100 rounded-tl-sm shadow-sm"
            }`}
        >
          {message.pending && !message.content ? (
            <span className="flex gap-1 py-1">
              <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-current [animation-delay:0ms]" />
              <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-current [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-current [animation-delay:300ms]" />
            </span>
          ) : (
            <p>{message.content}</p>
          )}
        </div>
      </div>
    </div>
  );
}
