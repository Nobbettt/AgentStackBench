import { expect, test } from "vitest";
import { parseGithubUrl } from "../../src/utils/parse-github-url";

test("parseGithubUrl", () => {
  expect(
    parseGithubUrl("https://github.com/alexreardon/tiny-invariant"),
  ).toEqual(["alexreardon", "tiny-invariant"]);

  expect(parseGithubUrl("https://github.com/anonymous/OpenHands")).toEqual([
    "All-Hands-AI",
    "OpenHands",
  ]);

  expect(parseGithubUrl("https://github.com/anonymous/")).toEqual([
    "All-Hands-AI",
    "",
  ]);

  expect(parseGithubUrl("https://github.com/")).toEqual([]);
});
