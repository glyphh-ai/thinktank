// Dev-friendly email: console by default; Resend (HTTP API) in prod.

import { config } from "./config";

export interface EmailMessage { to: string; subject: string; body: string }

export async function sendEmail(msg: EmailMessage): Promise<void> {
  if (config.emailBackend === "resend" && config.resendApiKey) {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { authorization: `Bearer ${config.resendApiKey}`, "content-type": "application/json" },
      body: JSON.stringify({ from: config.emailFrom, to: [msg.to], subject: msg.subject, text: msg.body }),
    });
    if (!res.ok) throw new Error(`Resend failed (${res.status}): ${await res.text()}`);
    return;
  }
  // console backend (dev): the PIN/link is visible in the server log.
  console.log(`\n=== EMAIL ===\nTo: ${msg.to}\nSubject: ${msg.subject}\n${msg.body}\n=============\n`);
}

export const verificationPinEmail = (to: string, pin: string): EmailMessage => ({
  to, subject: "Your plot-hole-finder verification code",
  body: `Your verification code is: ${pin}\n\nIt expires in ${Math.round(config.emailPinTtlSeconds / 60)} minutes.`,
});

export const passwordResetEmail = (to: string, url: string): EmailMessage => ({
  to, subject: "Reset your plot-hole-finder password",
  body: `Reset your password: ${url}\n\nThis link expires in ${Math.round(config.passwordResetTtlSeconds / 60)} minutes.`,
});
