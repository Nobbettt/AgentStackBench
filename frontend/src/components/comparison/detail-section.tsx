
import type { DetailVariant } from "@/components/comparison/types";
import { TableHead } from "@/components/ui/table";
import { HelpIcon } from "@/components/comparison/shared";

export function DetailSection({
  title,
  variants,
  render,
}: {
  title: string;
  variants: DetailVariant[];
  render: (variant: DetailVariant) => React.ReactNode;
}) {
  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <div className={variants.length > 1 ? "grid gap-6 xl:grid-cols-2" : "grid gap-6"}>
        {variants.map((variant) => (
          <div key={variant.label} className="space-y-4">
            {render(variant)}
          </div>
        ))}
      </div>
    </section>
  );
}

export function TrajectoryTableHead({ label, explanation }: { label: string; explanation: string }) {
  return (
    <TableHead>
      <div className="flex items-center gap-2">
        <span>{label}</span>
        <HelpIcon label={label} explanation={explanation} />
      </div>
    </TableHead>
  );
}
