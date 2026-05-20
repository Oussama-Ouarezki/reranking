import { create } from "zustand";
import type { ChatMessage, ModelName } from "./types";

interface AppState {
  model: ModelName;
  topK: number;
  generate: boolean;
  messages: ChatMessage[];
  setModel: (m: ModelName) => void;
  setTopK: (k: number) => void;
  setGenerate: (g: boolean) => void;
  appendMessage: (m: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  clearMessages: () => void;
}

export const useApp = create<AppState>((set) => ({
  model: "monot5",
  topK: 10,
  generate: true,
  messages: [],
  setModel: (m) => set({ model: m }),
  setTopK: (k) => set({ topK: k }),
  setGenerate: (g) => set({ generate: g }),
  appendMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
  updateMessage: (id, patch) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
    })),
  clearMessages: () => set({ messages: [] }),
}));
