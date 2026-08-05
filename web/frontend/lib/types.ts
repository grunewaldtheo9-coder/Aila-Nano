export interface GenerationSettings {
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  top_p: number | null;
  repetition_penalty: number;
}

export const DEFAULT_SETTINGS: GenerationSettings = {
  max_new_tokens: 200,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.95,
  repetition_penalty: 1.15,
};

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  agent_type?: string | null;
  created_at?: number;
  pending?: boolean;
}

export interface AgentInfo {
  name: string;
  system_prompt: string;
}

export interface HealthInfo {
  status: string;
  model_loaded: boolean;
  num_parameters: number | null;
  vocab_size: number | null;
  device: string;
  agents: string[];
}

export interface RecallResult {
  id: number;
  content: string;
  score: number;
  combined_score: number;
  importance: number;
  created_at: number;
}

export const AGENT_LABELS: Record<string, string> = {
  general: "General Assistant",
  programming: "Programming Assistant",
  research: "Research Assistant",
  writing: "Writing Assistant",
};

export const AGENT_ICONS: Record<string, string> = {
  general: "💬",
  programming: "💻",
  research: "🔬",
  writing: "✍️",
};
