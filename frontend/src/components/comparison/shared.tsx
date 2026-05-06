
import { CircleHelp, Minus, TrendingDown, TrendingUp } from "lucide-react";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { deltaIndicatorClassName } from "@/components/comparison/format";
import type { ComparisonVariant, DeltaTone, MetricDirection } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

export function MetricDirectionBadge({ direction }: { direction: MetricDirection }) {
  const content =
    direction === "higher"
      ? { symbol: "↑", label: "Higher better" }
      : direction === "lower"
        ? { symbol: "↓", label: "Lower better" }
        : { symbol: "~", label: "Contextual" };

  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[10px] font-medium normal-case tracking-normal text-muted-foreground">
      <span>{content.symbol}</span>
      <span>{content.label}</span>
    </span>
  );
}

export function HelpIcon({ label, explanation }: { label: string; explanation: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex h-4 w-4 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground"
          aria-label={`What ${label} means`}
        >
          <CircleHelp className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{explanation}</TooltipContent>
    </Tooltip>
  );
}

export function DeltaIndicator({ label, delta, tone }: { label: string; delta: number; tone: DeltaTone }) {
  const Icon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus;
  return (
    <div className={cn("inline-flex items-center gap-1.5 text-sm font-medium tabular-nums", deltaIndicatorClassName(tone))}>
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </div>
  );
}

export function DeltaSectionLabel({
  baseline,
  treatment,
}: {
  baseline: ComparisonVariant;
  treatment: ComparisonVariant;
}) {
  return (
    <div className="text-right">
      <div className="text-sm font-medium text-muted-foreground">{treatment.name}</div>
      <div className="text-xs font-normal text-muted-foreground/80">Compared with {baseline.name}</div>
    </div>
  );
}

export function ComparisonSectionShell({
  title,
  children,
  headerAside,
  collapsible = false,
  defaultOpen = true,
}: {
  title: string;
  children: React.ReactNode;
  headerAside?: React.ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}) {
  if (!collapsible) {
    return (
      <section className="space-y-4">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          {headerAside}
        </div>
        {children}
      </section>
    );
  }

  const value = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  return (
    <section className="space-y-4">
      <Accordion
        type="single"
        collapsible
        defaultValue={defaultOpen ? value : undefined}
        className="w-full rounded-lg border bg-background px-6"
      >
        <AccordionItem value={value} className="!border-b-0">
          <AccordionTrigger className="text-xl font-semibold tracking-tight hover:no-underline">
            <div className="flex w-full items-center justify-between gap-4 pr-4 text-left">
              <span>{title}</span>
              {headerAside}
            </div>
          </AccordionTrigger>
          <AccordionContent className="pt-2">{children}</AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
}
