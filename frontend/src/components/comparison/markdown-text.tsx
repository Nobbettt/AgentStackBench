
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type ListItem = {
  depth: number;
  text: string;
};

const inlineTokenPattern = /`([^`\n]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*\n]+)\*|_([^_\n]+)_/g;

export function MarkdownText({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("space-y-4 break-words text-sm leading-6", className)}>
      {renderMarkdown(text)}
    </div>
  );
}

function renderMarkdown(markdown: string): ReactNode[] {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();
    const key = `markdown-block-${index}`;

    if (!trimmed) {
      index += 1;
      continue;
    }

    const fenceMatch = trimmed.match(/^```([A-Za-z0-9_-]+)?\s*$/);
    if (fenceMatch) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().match(/^```\s*$/)) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      blocks.push(
        <pre key={key} className="overflow-auto rounded-md border bg-muted/30 p-4 text-xs leading-6">
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      blocks.push(renderHeading(level, headingMatch[2], key));
      index += 1;
      continue;
    }

    if (isHorizontalRule(trimmed)) {
      blocks.push(<hr key={key} className="border-border" />);
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      const tableLines: string[] = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        tableLines.push(lines[index]);
        index += 1;
      }
      blocks.push(renderTable(tableLines, key));
      continue;
    }

    const unorderedItem = parseListItem(line, "unordered");
    if (unorderedItem) {
      const items: ListItem[] = [];
      while (index < lines.length) {
        const item = parseListItem(lines[index], "unordered");
        if (!item) break;
        items.push(item);
        index += 1;
      }
      blocks.push(renderList(items, "unordered", key));
      continue;
    }

    const orderedItem = parseListItem(line, "ordered");
    if (orderedItem) {
      const items: ListItem[] = [];
      while (index < lines.length) {
        const item = parseListItem(lines[index], "ordered");
        if (!item) break;
        items.push(item);
        index += 1;
      }
      blocks.push(renderList(items, "ordered", key));
      continue;
    }

    if (trimmed.startsWith(">")) {
      const quoteLines: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push(
        <blockquote key={key} className="border-l-4 border-muted-foreground/30 pl-4 text-muted-foreground">
          {renderInline(quoteLines.join(" "), key)}
        </blockquote>,
      );
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length && shouldContinueParagraph(lines, index)) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    blocks.push(
      <p key={key} className="text-foreground">
        {renderInline(paragraphLines.join(" "), key)}
      </p>,
    );
  }

  return blocks.length > 0 ? blocks : [<p key="markdown-empty" className="text-muted-foreground">No content recorded.</p>];
}

function shouldContinueParagraph(lines: string[], index: number): boolean {
  const trimmed = lines[index].trim();
  if (!trimmed) return false;
  if (trimmed.match(/^```([A-Za-z0-9_-]+)?\s*$/)) return false;
  if (trimmed.match(/^(#{1,4})\s+.+$/)) return false;
  if (isHorizontalRule(trimmed)) return false;
  if (isTableStart(lines, index)) return false;
  if (parseListItem(lines[index], "unordered") || parseListItem(lines[index], "ordered")) return false;
  if (trimmed.startsWith(">")) return false;
  return true;
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let tokenIndex = 0;

  inlineTokenPattern.lastIndex = 0;
  for (const match of text.matchAll(inlineTokenPattern)) {
    if (match.index === undefined) continue;
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    const key = `${keyPrefix}-inline-${tokenIndex}`;
    if (match[1]) {
      nodes.push(
        <code key={key} className="rounded bg-muted px-1 py-0.5 text-[0.85em]">
          {match[1]}
        </code>,
      );
    } else if (match[2] && match[3]) {
      nodes.push(
        <a key={key} href={match[3]} target="_blank" rel="noreferrer" className="font-medium text-primary underline underline-offset-2">
          {match[2]}
        </a>,
      );
    } else if (match[4] || match[5]) {
      nodes.push(<strong key={key} className="font-semibold">{match[4] ?? match[5]}</strong>);
    } else if (match[6] || match[7]) {
      nodes.push(<em key={key}>{match[6] ?? match[7]}</em>);
    }

    lastIndex = match.index + match[0].length;
    tokenIndex += 1;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function parseListItem(line: string, type: "ordered" | "unordered"): ListItem | null {
  const match =
    type === "ordered"
      ? line.match(/^(\s*)\d+[.)]\s+(.+)$/)
      : line.match(/^(\s*)[-*+]\s+(.+)$/);
  if (!match) return null;
  return {
    depth: Math.floor(match[1].replace(/\t/g, "  ").length / 2),
    text: match[2],
  };
}

function renderList(items: ListItem[], type: "ordered" | "unordered", key: string) {
  const List = type === "ordered" ? "ol" : "ul";
  return (
    <List key={key} className={cn("space-y-1 pl-5", type === "ordered" ? "list-decimal" : "list-disc")}>
      {items.map((item, itemIndex) => (
        <li key={`${key}-${itemIndex}`} style={item.depth > 0 ? { marginLeft: `${item.depth}rem` } : undefined}>
          {renderInline(item.text, `${key}-${itemIndex}`)}
        </li>
      ))}
    </List>
  );
}

function isTableStart(lines: string[], index: number): boolean {
  if (index + 1 >= lines.length) return false;
  const headerCells = splitTableRow(lines[index]);
  const separatorCells = splitTableRow(lines[index + 1]);
  return headerCells.length > 1 && separatorCells.length === headerCells.length && separatorCells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderTable(lines: string[], key: string) {
  const header = splitTableRow(lines[0]);
  const rows = lines.slice(2).map(splitTableRow).filter((row) => row.length === header.length);
  return (
    <div key={key} className="overflow-x-auto rounded-md border">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-muted/40">
          <tr>
            {header.map((cell, cellIndex) => (
              <th key={`${key}-head-${cellIndex}`} className="border-b px-3 py-2 font-semibold">
                {renderInline(cell, `${key}-head-${cellIndex}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${key}-row-${rowIndex}`} className="border-b last:border-b-0">
              {row.map((cell, cellIndex) => (
                <td key={`${key}-row-${rowIndex}-${cellIndex}`} className="px-3 py-2 align-top">
                  {renderInline(cell, `${key}-row-${rowIndex}-${cellIndex}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isHorizontalRule(line: string): boolean {
  return /^([-*_])(?:\s*\1){2,}$/.test(line);
}

function headingClassName(level: number): string {
  if (level === 1) return "pt-1 text-lg font-semibold tracking-tight";
  if (level === 2) return "pt-1 text-base font-semibold tracking-tight";
  return "text-sm font-semibold tracking-tight";
}

function renderHeading(level: number, text: string, key: string) {
  const children = renderInline(text, key);
  const className = headingClassName(level);
  if (level === 1) return <h4 key={key} className={className}>{children}</h4>;
  if (level === 2) return <h5 key={key} className={className}>{children}</h5>;
  return <h6 key={key} className={className}>{children}</h6>;
}
