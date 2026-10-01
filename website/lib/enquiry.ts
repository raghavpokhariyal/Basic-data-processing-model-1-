import { enquiry } from './content';

export interface EnquiryInput {
  name: string;
  phone: string;
  email: string;
  service: string;
  budget: string;
  message: string;
  /** Honeypot: real visitors leave this empty */
  company?: string;
}

export type EnquiryErrors = Partial<Record<keyof EnquiryInput, string>>;

const INDIAN_MOBILE = /^(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}$/;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

/** Shared by the form (instant feedback) and the API route (authoritative). */
export function validateEnquiry(input: Partial<EnquiryInput>): EnquiryErrors {
  const e: EnquiryErrors = {};
  const name = (input.name ?? '').trim();
  const phone = (input.phone ?? '').trim();
  const email = (input.email ?? '').trim();
  const message = (input.message ?? '').trim();
  if (name.length < 2) e.name = 'Please tell us your name.';
  else if (name.length > 80) e.name = 'That name is a little long.';
  if (!INDIAN_MOBILE.test(phone)) e.phone = 'Enter a 10-digit Indian mobile number.';
  if (!EMAIL.test(email) || email.length > 120) e.email = 'Enter a valid email address.';
  if (!enquiry.serviceOptions.includes(input.service ?? '')) e.service = 'Choose the service you need.';
  if (!enquiry.budgetOptions.includes(input.budget ?? '')) e.budget = 'Choose an approximate budget.';
  if (message.length > 2000) e.message = 'Please keep your message under 2,000 characters.';
  return e;
}
