import type { Theme } from "../hooks/useTheme";

const AGENT_PALETTES: Record<Theme, string[]> = {
  light: [
    "#374151", "#1e3a5f", "#5b4a6a", "#4a5240", "#7c4a3a",
    "#2d5a5a", "#5a4630", "#3b4f7a", "#6b4050", "#2e5e4e",
  ],
  dark: [
    "#b4bcc9", "#7fa5d2", "#b598c3", "#9aaa82", "#ce8d73",
    "#70aaa6", "#bc986b", "#879fce", "#c7889b", "#75aa93",
  ],
};

export function agentColor(index: number, theme: Theme): string {
  const palette = AGENT_PALETTES[theme];
  if (index < palette.length) return palette[index];
  const hue = ((index - palette.length) * 137.5) % 360;
  return `hsl(${hue}, 35%, ${theme === "dark" ? 68 : 40}%)`;
}
