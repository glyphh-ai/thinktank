export const metadata = { title: "Semantic Meters — Glyphh" };

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#0b0e14", color: "#e6e6e6", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
        {children}
      </body>
    </html>
  );
}
