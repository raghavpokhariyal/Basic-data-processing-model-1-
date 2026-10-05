const paths: Record<string, React.ReactNode> = {
  Instagram: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="5" />
      <circle cx="12" cy="12" r="4" />
      <circle cx="17.5" cy="6.5" r="0.9" fill="currentColor" stroke="none" />
    </>
  ),
  Pinterest: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M10.5 20l1.6-7M12.1 13c.4 1.2 1.4 1.8 2.5 1.8 2 0 3.4-1.9 3.4-4.3 0-2.6-2.2-4.5-5.2-4.5-3.3 0-5.3 2.3-5.3 4.7 0 1.2.5 2.3 1.3 2.7" />
    </>
  ),
  LinkedIn: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      <path d="M8 10.5V17M8 7.5v.01M12 17v-4c0-1.5 1-2.5 2.2-2.5S16 11.5 16 13v4M12 10.5V17" />
    </>
  ),
  YouTube: (
    <>
      <rect x="2.5" y="5.5" width="19" height="13" rx="4" />
      <path d="M10.5 9.5v5l4.2-2.5z" fill="currentColor" />
    </>
  ),
  Facebook: <path d="M14 21v-7.5h2.6l.4-3H14V8.7c0-.9.3-1.5 1.6-1.5H17V4.5c-.3 0-1.2-.1-2.3-.1-2.3 0-3.9 1.4-3.9 4v2.1H8.3v3h2.5V21" />,
  WhatsApp: (
    <>
      <path d="M4 20l1.2-3.6A8 8 0 1 1 8 19.1z" />
      <path d="M9.2 8.6c.2-.4.4-.4.7-.4h.5c.2 0 .4.1.5.4l.7 1.6c.1.2 0 .4-.1.6l-.5.6c-.1.1-.1.3 0 .5.6 1 1.4 1.8 2.5 2.4.2.1.4.1.5-.1l.6-.7c.2-.2.4-.2.6-.1l1.5.7c.3.1.4.3.3.6-.2.9-1 1.5-1.9 1.5-2.7-.2-5.6-2.9-5.9-5.6 0-.7.2-1.4.5-1.9z" fill="currentColor" stroke="none" />
    </>
  ),
};

export function SocialIcon({ name, className = 'h-5 w-5' }: { name: string; className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {paths[name]}
    </svg>
  );
}
