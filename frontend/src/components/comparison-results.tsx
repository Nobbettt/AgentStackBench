
import type { ComparisonCard } from "@/data/comparisons";
import { TooltipProvider } from "@/components/ui/tooltip";
import { InstanceResultsSection } from "@/components/comparison/instance-results-section";
import { ComparisonInstanceDetailPage } from "@/components/comparison/instance-detail-page";
import { ComparisonMetricSections, OutcomeBreakdownSection } from "@/components/comparison/metric-sections";
import { SkillUsageSection, ToolUsageSection } from "@/components/comparison/usage-sections";
import type { ComparisonResultsViewMode, DeltaDisplayMode } from "@/components/comparison/types";

export { ComparisonInstanceDetailPage };

export function ComparisonResults({
  comparison,
  viewMode,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  return (
    <TooltipProvider>
      <div className="space-y-6">
        <OutcomeBreakdownSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible />
        <ComparisonMetricSections comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible />
        <SkillUsageSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible />
        <ToolUsageSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible />
        <InstanceResultsSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
      </div>
    </TooltipProvider>
  );
}
