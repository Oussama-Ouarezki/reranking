// Per-model color tokens. Each model gets a Tailwind palette so it's
// visually distinct in sidebars, tables, and labels.
//
// `chip`   — small badge: bg + text + border
// `dot`    — solid colored dot for legend / inline marker
// `bar`    — bg used as an accent bar / row tint
// `text`   — text color for the model label

export interface ModelColor {
  chip: string;
  dot: string;
  bar: string;
  text: string;
}

const PALETTE: ModelColor[] = [
  { chip: "bg-slate-100 text-slate-800 border-slate-300", dot: "bg-slate-500", bar: "bg-slate-400", text: "text-slate-700" },
  { chip: "bg-blue-100 text-blue-900 border-blue-200", dot: "bg-blue-500", bar: "bg-blue-400", text: "text-blue-700" },
  { chip: "bg-cyan-100 text-cyan-900 border-cyan-200", dot: "bg-cyan-500", bar: "bg-cyan-400", text: "text-cyan-700" },
  { chip: "bg-indigo-100 text-indigo-900 border-indigo-200", dot: "bg-indigo-500", bar: "bg-indigo-400", text: "text-indigo-700" },
  { chip: "bg-purple-100 text-purple-900 border-purple-200", dot: "bg-purple-500", bar: "bg-purple-400", text: "text-purple-700" },
  { chip: "bg-fuchsia-100 text-fuchsia-900 border-fuchsia-200", dot: "bg-fuchsia-500", bar: "bg-fuchsia-400", text: "text-fuchsia-700" },
  { chip: "bg-pink-100 text-pink-900 border-pink-200", dot: "bg-pink-500", bar: "bg-pink-400", text: "text-pink-700" },
  { chip: "bg-rose-100 text-rose-900 border-rose-200", dot: "bg-rose-500", bar: "bg-rose-400", text: "text-rose-700" },
  { chip: "bg-amber-100 text-amber-900 border-amber-200", dot: "bg-amber-500", bar: "bg-amber-400", text: "text-amber-700" },
  { chip: "bg-orange-100 text-orange-900 border-orange-200", dot: "bg-orange-500", bar: "bg-orange-400", text: "text-orange-700" },
  { chip: "bg-yellow-100 text-yellow-900 border-yellow-200", dot: "bg-yellow-500", bar: "bg-yellow-400", text: "text-yellow-700" },
  { chip: "bg-lime-100 text-lime-900 border-lime-200", dot: "bg-lime-500", bar: "bg-lime-400", text: "text-lime-700" },
  { chip: "bg-green-100 text-green-900 border-green-200", dot: "bg-green-500", bar: "bg-green-400", text: "text-green-700" },
  { chip: "bg-emerald-100 text-emerald-900 border-emerald-200", dot: "bg-emerald-500", bar: "bg-emerald-400", text: "text-emerald-700" },
  { chip: "bg-teal-100 text-teal-900 border-teal-200", dot: "bg-teal-500", bar: "bg-teal-400", text: "text-teal-700" },
  { chip: "bg-sky-100 text-sky-900 border-sky-200", dot: "bg-sky-500", bar: "bg-sky-400", text: "text-sky-700" },
  { chip: "bg-violet-100 text-violet-900 border-violet-200", dot: "bg-violet-500", bar: "bg-violet-400", text: "text-violet-700" },
  { chip: "bg-red-100 text-red-900 border-red-200", dot: "bg-red-500", bar: "bg-red-400", text: "text-red-700" },
];

const FALLBACK: ModelColor = {
  chip: "bg-bg text-ink border-border",
  dot: "bg-muted",
  bar: "bg-muted",
  text: "text-muted",
};

// Stable hash → palette index so the same model name always picks the same color
// even if new models are added or the list reorders.
function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function modelColor(model?: string | null): ModelColor {
  if (!model) return FALLBACK;
  return PALETTE[hashStr(model) % PALETTE.length];
}
