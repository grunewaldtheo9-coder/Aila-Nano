import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aila Nano",
  description: "Chat with Aila Nano — a small language model by Aila Company Solutions.",
};

// Runs before React hydrates so the correct theme class is present on
// first paint (no flash-of-wrong-theme).
const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = localStorage.getItem('aila-theme');
    var theme = stored || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    if (theme === 'dark') document.documentElement.classList.add('dark');
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="h-screen overflow-hidden antialiased">{children}</body>
    </html>
  );
}
