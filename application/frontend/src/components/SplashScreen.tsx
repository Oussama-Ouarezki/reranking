import { useEffect, useState } from "react";

export default function SplashScreen({ onDone }: { onDone: () => void }) {
  const [phase, setPhase] = useState<"enter" | "hold" | "exit">("enter");

  useEffect(() => {
    const t1 = setTimeout(() => setPhase("hold"), 800);
    const t2 = setTimeout(() => setPhase("exit"), 1800);
    const t3 = setTimeout(onDone, 2600);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [onDone]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        backgroundColor: "#6b7280",
        opacity: phase === "exit" ? 0 : 1,
        transition: phase === "exit" ? "opacity 0.7s ease-in-out" : undefined,
        pointerEvents: phase === "exit" ? "none" : "all",
      }}
    >
      <div
        style={{
          transform: phase === "enter" ? "translateY(-60px)" : "translateY(0)",
          opacity: phase === "enter" ? 0 : 1,
          transition: "transform 0.7s cubic-bezier(0.22,1,0.36,1), opacity 0.5s ease-out",
        }}
        className="text-center select-none"
      >
        <span
          className="font-bold tracking-widest text-white"
          style={{ fontSize: "5rem", letterSpacing: "0.25em", fontFamily: "monospace" }}
        >
          RAG
        </span>
        <p className="mt-2 text-white/70 text-sm tracking-widest uppercase">
          Biomedical Retrieval-Augmented Generation
        </p>
      </div>
    </div>
  );
}
