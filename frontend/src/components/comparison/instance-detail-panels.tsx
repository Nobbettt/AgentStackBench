// SPDX-License-Identifier: Apache-2.0

import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import type { ComparisonInstanceDetail } from "@/data/comparisons";
import { DetailSection } from "@/components/comparison/detail-section";
import { MarkdownText } from "@/components/comparison/markdown-text";
import { cn } from "@/lib/utils";

export function FinalAnswerSection({ variants }: { variants: ComparisonInstanceDetail["variants"] }) {
  return (
    <DetailSection title="Final Answer" variants={variants} render={(variant) => (
      <MarkdownTextPanel key={variant.label} title={variant.name} text={variant.finalOutput?.finalAnswer || "No final answer recorded."} />
    )} />
  );
}

export function ModelPatchSection({ variants }: { variants: ComparisonInstanceDetail["variants"] }) {
  if (!variants.some((variant) => variant.modelPatch)) {
    return <StatusPanel message="No model patch was recorded for this instance." />;
  }
  if (variants.length === 2) {
    return <ResizableModelPatchComparison left={variants[0]} right={variants[1]} />;
  }

  return (
    <DetailSection title="Model Patch" variants={variants} render={(variant) => (
      <DiffCodePanel key={variant.label} title={variant.name} text={variant.modelPatch || "No model patch recorded for this run."} />
    )} />
  );
}

function StatusPanel({ message }: { message: string }) {
  return (
    <section className="rounded-lg bg-background p-6 text-sm text-muted-foreground">
      {message}
    </section>
  );
}

function ResizableModelPatchComparison({
  left,
  right,
}: {
  left: ComparisonInstanceDetail["variants"][number];
  right: ComparisonInstanceDetail["variants"][number];
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [leftPercent, setLeftPercent] = useState(50);

  const updateLeftPercent = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const nextPercent = ((clientX - rect.left) / rect.width) * 100;
    setLeftPercent(Math.min(78, Math.max(22, nextPercent)));
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    updateLeftPercent(event.clientX);

    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const handlePointerMove = (moveEvent: globalThis.PointerEvent) => {
      updateLeftPercent(moveEvent.clientX);
    };
    const handlePointerUp = () => {
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight" && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    if (event.key === "Home") {
      setLeftPercent(22);
      return;
    }
    if (event.key === "End") {
      setLeftPercent(78);
      return;
    }
    setLeftPercent((current) => {
      const delta = event.key === "ArrowLeft" ? -4 : 4;
      return Math.min(78, Math.max(22, current + delta));
    });
  };

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">Model Patch</h2>
      <div ref={containerRef} className="flex min-h-[32rem] w-full min-w-0 overflow-hidden rounded-lg">
        <div className="min-w-0" style={{ flex: `0 0 ${leftPercent}%` }}>
          <DiffCodePanel
            title={left.name}
            text={left.modelPatch || "No model patch recorded for this run."}
            className="rounded-r-none pr-0"
            preClassName="max-h-[42rem] rounded-r-none"
          />
        </div>
        <div
          className="group flex w-4 shrink-0 cursor-col-resize items-stretch justify-center bg-muted/20 outline-none transition-colors hover:bg-muted/50 focus-visible:bg-muted"
          role="separator"
          aria-label="Resize model patch comparison panes"
          aria-orientation="vertical"
          aria-valuemin={22}
          aria-valuemax={78}
          aria-valuenow={Math.round(leftPercent)}
          tabIndex={0}
          onPointerDown={handlePointerDown}
          onKeyDown={handleKeyDown}
        >
          <div className="my-4 w-1 rounded-full bg-border transition-colors group-hover:bg-muted-foreground/50" />
        </div>
        <div className="min-w-0 flex-1">
          <DiffCodePanel
            title={right.name}
            text={right.modelPatch || "No model patch recorded for this run."}
            className="rounded-l-none pl-0"
            titleClassName="pl-5"
            preClassName="max-h-[42rem] rounded-l-none"
          />
        </div>
      </div>
    </section>
  );
}

function MarkdownTextPanel({ title, text }: { title: string; text: string }) {
  return (
    <div className="h-full rounded-lg bg-background p-5">
      <h3 className="text-lg font-semibold">{title}</h3>
      <MarkdownText text={text} className="mt-4" />
    </div>
  );
}

function DiffCodePanel({
  title,
  text,
  className,
  titleClassName,
  preClassName,
}: {
  title: string;
  text: string;
  className?: string;
  titleClassName?: string;
  preClassName?: string;
}) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");

  return (
    <div className={cn("h-full rounded-lg bg-background p-5", className)}>
      <h3 className={cn("text-lg font-semibold", titleClassName)}>{title}</h3>
      <pre className={cn("mt-4 max-h-[28rem] overflow-auto rounded-md border bg-muted/20 py-3 text-xs leading-6", preClassName)}>
        <code>
          {lines.map((line, index) => (
            <span
              key={`${title}-patch-line-${index}`}
              className={cn("block min-h-6 whitespace-pre px-4", diffLineClassName(line))}
            >
              {line || " "}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

function diffLineClassName(line: string): string {
  if (line.startsWith("diff --git ") || line.startsWith("index ")) {
    return "bg-muted/50 font-semibold text-muted-foreground";
  }
  if (line.startsWith("@@")) {
    return "border-l-2 border-sky-500 bg-sky-50 text-sky-900";
  }
  if (line.startsWith("+++") || line.startsWith("---")) {
    return "bg-muted/40 font-medium text-muted-foreground";
  }
  if (line.startsWith("+")) {
    return "border-l-2 border-emerald-500 bg-emerald-50 text-emerald-950";
  }
  if (line.startsWith("-")) {
    return "border-l-2 border-rose-500 bg-rose-50 text-rose-950";
  }
  if (line.startsWith("\\ No newline at end of file")) {
    return "text-muted-foreground";
  }
  return "";
}
