"use client";

import { GenerationSettings } from "@/lib/types";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: GenerationSettings;
  onChange: (s: GenerationSettings) => void;
  streaming: boolean;
  onStreamingChange: (v: boolean) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>{label}</span>
        <span className="font-mono">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-aila-500"
      />
    </label>
  );
}

export default function SettingsPanel({
  open,
  onClose,
  settings,
  onChange,
  streaming,
  onStreamingChange,
}: Props) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onClick={onClose}>
      <div
        className="h-full w-80 max-w-full overflow-y-auto bg-white p-5 shadow-xl dark:bg-gray-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Generation settings</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
            ✕
          </button>
        </div>

        <div className="space-y-5">
          <label className="flex items-center justify-between text-sm">
            <span>Stream responses</span>
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => onStreamingChange(e.target.checked)}
              className="h-4 w-4 accent-aila-500"
            />
          </label>

          <Slider
            label="Temperature"
            value={settings.temperature}
            min={0}
            max={2}
            step={0.05}
            onChange={(v) => onChange({ ...settings, temperature: v })}
          />
          <Slider
            label="Top-p"
            value={settings.top_p ?? 0.95}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => onChange({ ...settings, top_p: v })}
          />
          <Slider
            label="Top-k"
            value={settings.top_k ?? 40}
            min={1}
            max={200}
            step={1}
            onChange={(v) => onChange({ ...settings, top_k: Math.round(v) })}
          />
          <Slider
            label="Repetition penalty"
            value={settings.repetition_penalty}
            min={1}
            max={2}
            step={0.05}
            onChange={(v) => onChange({ ...settings, repetition_penalty: v })}
          />
          <Slider
            label="Max new tokens"
            value={settings.max_new_tokens}
            min={16}
            max={512}
            step={8}
            onChange={(v) => onChange({ ...settings, max_new_tokens: Math.round(v) })}
          />
        </div>

        <p className="mt-6 text-xs leading-relaxed text-gray-400">
          These control how Aila Nano samples the next token during generation. Lower
          temperature/top-p make responses more focused and deterministic; higher values make
          them more varied and creative.
        </p>
      </div>
    </div>
  );
}
