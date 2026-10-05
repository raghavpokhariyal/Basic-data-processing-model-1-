import { NextResponse } from 'next/server';
import { validateEnquiry, type EnquiryInput } from '@/lib/enquiry';

/**
 * Receives enquiries from the contact form.
 *
 * Set ENQUIRY_WEBHOOK_URL (Zapier, Make, Slack, Google Apps Script, a CRM…)
 * to forward each enquiry as JSON. Without it, enquiries are only logged
 * on the server, so wire this up before launch.
 */

const recent = new Map<string, number[]>();
const WINDOW_MS = 10 * 60 * 1000;
const MAX_PER_WINDOW = 5;

function rateLimited(ip: string) {
  const now = Date.now();
  const hits = (recent.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  hits.push(now);
  recent.set(ip, hits);
  return hits.length > MAX_PER_WINDOW;
}

export async function POST(request: Request) {
  let body: Partial<EnquiryInput>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid request.' }, { status: 400 });
  }

  // Honeypot filled in: pretend success, drop silently
  if (body.company) return NextResponse.json({ ok: true });

  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? 'unknown';
  if (rateLimited(ip)) {
    return NextResponse.json({ ok: false, error: 'Too many enquiries. Please try again later or reach us on WhatsApp.' }, { status: 429 });
  }

  const errors = validateEnquiry(body);
  if (Object.keys(errors).length) return NextResponse.json({ ok: false, errors }, { status: 422 });

  const enquiry = {
    name: body.name!.trim(),
    phone: body.phone!.trim(),
    email: body.email!.trim(),
    service: body.service,
    budget: body.budget,
    message: (body.message ?? '').trim(),
    receivedAt: new Date().toISOString(),
  };

  const webhook = process.env.ENQUIRY_WEBHOOK_URL;
  if (webhook) {
    try {
      const res = await fetch(webhook, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(enquiry) });
      if (!res.ok) throw new Error(`Webhook responded ${res.status}`);
    } catch (err) {
      console.error('[enquiry] forwarding failed', err);
      return NextResponse.json({ ok: false, error: 'We could not send your enquiry just now. Please try WhatsApp or call us.' }, { status: 502 });
    }
  } else {
    console.info('[enquiry] received (set ENQUIRY_WEBHOOK_URL to forward):', enquiry);
  }

  return NextResponse.json({ ok: true });
}
