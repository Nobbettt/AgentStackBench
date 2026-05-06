
import * as React from "react";
import * as RechartsPrimitive from "recharts";

import { cn } from "@/lib/utils";

export type ChartConfig = Record<
  string,
  {
    label?: React.ReactNode;
    color?: string;
  }
>;

type ChartContextValue = {
  config: ChartConfig;
};

const ChartContext = React.createContext<ChartContextValue | null>(null);

function useChart() {
  const context = React.useContext(ChartContext);
  if (!context) {
    throw new Error("Chart components must be used within a ChartContainer.");
  }
  return context;
}

function resolveChartKey(item: any, fallbackKey?: string) {
  if (fallbackKey && item?.payload?.[fallbackKey] != null) {
    return String(item.payload[fallbackKey]);
  }
  if (item?.dataKey != null) {
    return String(item.dataKey);
  }
  if (item?.name != null) {
    return String(item.name);
  }
  if (item?.value != null) {
    return String(item.value);
  }
  return undefined;
}

export const ChartContainer = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    config: ChartConfig;
  }
>(({ config, className, style, children, ...props }, ref) => {
  const colorVariables = Object.fromEntries(
    Object.entries(config).flatMap(([key, value]) =>
      value.color ? [[`--color-${key}`, value.color]] : [],
    ),
  ) as React.CSSProperties;

  return (
    <ChartContext.Provider value={{ config }}>
      <div
        ref={ref}
        className={cn(
          "flex w-full items-stretch justify-center text-xs",
          "[&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground",
          "[&_.recharts-cartesian-grid_line]:stroke-border/60",
          "[&_.recharts-layer.recharts-cartesian-axis-line]:stroke-border/60",
          className,
        )}
        style={{ ...colorVariables, ...style }}
        {...props}
      >
        {children}
      </div>
    </ChartContext.Provider>
  );
});
ChartContainer.displayName = "ChartContainer";

export const ChartTooltip = RechartsPrimitive.Tooltip;
export const ChartLegend = RechartsPrimitive.Legend;

export const ChartTooltipContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    active?: boolean;
    payload?: any[];
    label?: React.ReactNode;
    hideLabel?: boolean;
    hideIndicator?: boolean;
    indicator?: "dot" | "line" | "dashed";
    labelFormatter?: (value: React.ReactNode, payload: any[]) => React.ReactNode;
    formatter?: (value: any, name: string, item: any, index: number) => React.ReactNode;
    labelKey?: string;
    nameKey?: string;
  }
>(
  (
    {
      active,
      payload,
      label,
      className,
      hideLabel = false,
      hideIndicator = false,
      indicator = "dot",
      labelFormatter,
      formatter,
      labelKey,
      nameKey,
      ...props
    },
    ref,
  ) => {
    const { config } = useChart();

    if (!active || !payload?.length) {
      return null;
    }

    const tooltipLabel =
      labelFormatter?.(
        labelKey && payload[0]?.payload?.[labelKey] != null ? payload[0].payload[labelKey] : label,
        payload,
      ) ??
      (labelKey && payload[0]?.payload?.[labelKey] != null ? payload[0].payload[labelKey] : label);

    return (
      <div
        ref={ref}
        className={cn("grid min-w-[12rem] gap-2 rounded-lg border bg-background px-3 py-2 text-xs shadow-xl", className)}
        {...props}
      >
        {!hideLabel && tooltipLabel != null ? <div className="font-medium text-foreground">{tooltipLabel}</div> : null}
        <div className="grid gap-2">
          {payload.map((item, index) => {
            const chartKey = resolveChartKey(item, nameKey);
            const chartConfig = chartKey ? config[chartKey] : undefined;
            const displayName = chartConfig?.label ?? item.name ?? chartKey ?? "Value";
            const displayValue =
              formatter?.(item.value, String(displayName), item, index) ??
              (item.value !== undefined && item.value !== null ? item.value.toLocaleString?.() ?? String(item.value) : "—");
            const indicatorColor = item.color ?? item.payload?.fill ?? "hsl(var(--foreground))";

            return (
              <div key={`${chartKey ?? displayName}-${index}`} className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-muted-foreground">
                  {!hideIndicator ? (
                    indicator === "line" ? (
                      <span className="h-0.5 w-3 rounded-full" style={{ backgroundColor: indicatorColor }} />
                    ) : indicator === "dashed" ? (
                      <span
                        className="h-0.5 w-3 border-t border-dashed"
                        style={{ borderColor: indicatorColor }}
                      />
                    ) : (
                      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: indicatorColor }} />
                    )
                  ) : null}
                  <span>{displayName}</span>
                </div>
                <span className="font-mono font-medium tabular-nums text-foreground">{displayValue}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  },
);
ChartTooltipContent.displayName = "ChartTooltipContent";

export const ChartLegendContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    payload?: any[];
    nameKey?: string;
  }
>(({ className, payload, nameKey, ...props }, ref) => {
  const { config } = useChart();

  if (!payload?.length) {
    return null;
  }

  return (
    <div ref={ref} className={cn("flex flex-wrap items-center gap-4 text-sm", className)} {...props}>
      {payload.map((item, index) => {
        const chartKey = resolveChartKey(item, nameKey);
        const chartConfig = chartKey ? config[chartKey] : undefined;
        const label = chartConfig?.label ?? item.value ?? chartKey;
        const color = item.color ?? "hsl(var(--foreground))";

        return (
          <div key={`${chartKey ?? item.value ?? index}`} className="flex items-center gap-2 text-muted-foreground">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span>{label}</span>
          </div>
        );
      })}
    </div>
  );
});
ChartLegendContent.displayName = "ChartLegendContent";
