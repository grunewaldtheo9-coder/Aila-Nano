"use client";

import { useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onUpload: (file: File) => void;
  disabled?: boolean;
}

export default function Composer({ onSend, onUpload, disabled }: Props) {
  const [text, setText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <div className="border-t border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900 sm:p-4">
      <div className="flex items-end gap-2 rounded-2xl border border-gray-300 bg-gray-50 p-2 focus-within:border-aila-400 dark:border-gray-700 dark:bg-gray-800">
        <button
          onClick={() => fileInputRef.current?.click()}
          title="Upload a file to Aila Nano's knowledge base"
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.jsonl,.csv,.log"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Message Aila Nano..."
          rows={1}
          className="max-h-40 flex-1 resize-none bg-transparent px-1 py-1.5 text-sm outline-none placeholder:text-gray-400"
        />
        <button
          onClick={submit}
          disabled={disabled || !text.trim()}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-aila-500 text-white
            transition-colors hover:bg-aila-600 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700"
          aria-label="Send"
        >
          ➤
        </button>
      </div>
      <div className="mt-1.5 px-1 text-[11px] text-gray-400">
        Enter to send · Shift+Enter for a new line
      </div>
    </div>
  );
}
