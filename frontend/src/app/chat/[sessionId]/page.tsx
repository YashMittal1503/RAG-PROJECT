"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  getChatMessagesWithMetadata,
  deleteChatSession,
  sendQuery,
  updateChatSession,
} from "@/lib/api";
import {
  Send,
  ArrowUp,
  ArrowDown,
  Loader2,
  FileText,
  FileSpreadsheet,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Sparkles,
  Trash2,
  Pencil,
  Check,
  X,
  Pin,
  PinOff,
  Share2,
  Plus,
  Mic,
  Copy,
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Citation = {
  chunk_id: string;
  filename: string;
  page_number?: number;
  row_range_start?: number;
  row_range_end?: number;
  content_preview: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  sql_query?: string;
};

function formatMessageContent(content: string, citations?: Citation[]): string {
  if (!content) return "";

  // 1. Replace raw chunk UUIDs like 【CHUNK 650ff328...】 or [CHUNK 650ff328...] with [Page X]
  let formatted = content.replace(
    /(\[|【)\s*CHUNK\s+([a-f0-9\-]+)\s*(\]|】)/gi,
    (_match, _p1, chunkId) => {
      const cit = citations?.find((c) => c.chunk_id === chunkId);
      if (cit?.page_number) {
        return ` **[Page ${cit.page_number}]** `;
      }
      if (cit?.row_range_start) {
        return ` **[Rows ${cit.row_range_start}-${cit.row_range_end || "?"}]** `;
      }
      if (cit?.filename) {
        return ` **[${cit.filename}]** `;
      }
      if (citations && citations.length > 0 && citations[0].page_number) {
        return ` **[Page ${citations[0].page_number}]** `;
      }
      return "";
    }
  );

  // 2. Remove standalone CHUNK UUIDs without brackets
  formatted = formatted.replace(
    /CHUNK\s+[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/gi,
    ""
  );

  // 3. Normalize 【Page X】 or [Page X] or (Page X) to bold markdown badge **[Page X]**
  formatted = formatted.replace(
    /(\[|【)\s*(Page\s+\d+|Rows?\s+[\d\-]+|Summary)\s*(\]|】)/gi,
    " **[$2]** "
  );

  return formatted;
}

const markdownComponents = {
  h1: ({ children }: any) => (
    <h1 className="text-lg font-bold text-[var(--foreground)] mt-4 mb-2 first:mt-0 tracking-tight">
      {children}
    </h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-base font-bold text-[var(--foreground)] mt-3.5 mb-1.5 first:mt-0 tracking-tight">
      {children}
    </h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-sm font-bold text-[var(--foreground)] mt-3 mb-1 first:mt-0">
      {children}
    </h3>
  ),
  p: ({ children }: any) => (
    <p className="text-sm leading-relaxed mb-3 last:mb-0 text-[var(--foreground)]">
      {children}
    </p>
  ),
  ul: ({ children }: any) => (
    <ul className="list-disc pl-5 my-2 space-y-1.5 text-sm text-[var(--foreground)]">
      {children}
    </ul>
  ),
  ol: ({ children }: any) => (
    <ol className="list-decimal pl-5 my-2 space-y-1.5 text-sm text-[var(--foreground)]">
      {children}
    </ol>
  ),
  li: ({ children }: any) => (
    <li className="text-sm leading-relaxed text-[var(--foreground)]">
      {children}
    </li>
  ),
  strong: ({ children }: any) => {
    const text =
      typeof children === "string"
        ? children
        : Array.isArray(children) && typeof children[0] === "string"
        ? children[0]
        : "";
    const pageMatch = text.match(/^\[(Page\s+\d+|Rows?\s+[\d\-]+|Summary)\]$/i);
    if (pageMatch) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 mx-1 rounded-md text-xs font-semibold bg-[var(--primary)]/15 text-[var(--primary)] border border-[var(--primary)]/30 select-none align-middle shadow-xs">
          {pageMatch[1]}
        </span>
      );
    }
    return <strong className="font-bold text-[var(--foreground)]">{children}</strong>;
  },
  em: ({ children }: any) => (
    <em className="italic text-[var(--foreground)]/90">{children}</em>
  ),
  code: ({ inline, className, children, ...props }: any) => {
    const text = String(children).replace(/\n$/, "");
    const isShort = !text.includes("\n") && text.length < 60;

    if (inline || (!className && isShort)) {
      return (
        <code
          className="px-1.5 py-0.5 rounded-md bg-[var(--secondary)] text-[var(--primary)] font-mono text-xs border border-[var(--border)]"
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <div className="relative my-3 rounded-xl overflow-hidden border border-[var(--border)] bg-[#121214]">
        <pre className="p-3.5 overflow-x-auto font-mono text-xs text-slate-200">
          <code {...props}>{children}</code>
        </pre>
      </div>
    );
  },
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-2 border-[var(--primary)] pl-3.5 my-2.5 italic text-[var(--muted-foreground)]">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-4 border-[var(--border)]" />,
  table: ({ children }: any) => (
    <div className="my-3 w-full overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--secondary)]/30 shadow-xs">
      <table className="w-full text-left text-xs border-collapse min-w-full">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }: any) => (
    <thead className="bg-[var(--secondary)] text-[var(--foreground)] border-b border-[var(--border)] text-xs font-semibold">
      {children}
    </thead>
  ),
  tbody: ({ children }: any) => (
    <tbody className="divide-y divide-[var(--border)]/60 text-xs">
      {children}
    </tbody>
  ),
  tr: ({ children }: any) => (
    <tr className="transition-colors hover:bg-[var(--primary)]/5">
      {children}
    </tr>
  ),
  th: ({ children }: any) => (
    <th className="px-3.5 py-2.5 font-semibold text-[var(--foreground)] whitespace-nowrap text-left border-b border-[var(--border)]">
      {children}
    </th>
  ),
  td: ({ children }: any) => (
    <td className="px-3.5 py-2 text-[var(--foreground)]/90 whitespace-nowrap text-left">
      {children}
    </td>
  ),
};

function CitationChip({
  citation,
  index,
}: {
  citation: Citation;
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const isSpreadsheet =
    citation.row_range_start !== null && citation.row_range_start !== undefined;

  const label = isSpreadsheet
    ? `📊 Rows ${citation.row_range_start}-${citation.row_range_end || "?"} (${citation.filename})`
    : `📄 ${citation.page_number ? `Page ${citation.page_number}` : citation.filename}`;

  return (
    <div className="inline-block animate-fade-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-[var(--primary)]/15 text-[var(--primary)] hover:bg-[var(--primary)]/25 transition-colors border border-[var(--primary)]/30 shadow-xs cursor-pointer"
      >
        <span className="font-semibold">[{index + 1}]</span>
        <span className="max-w-[220px] truncate">{label}</span>
        {expanded ? (
          <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3" />
        )}
      </button>
      {expanded && (
        <div className="mt-2 p-3 rounded-xl bg-[var(--secondary)] border border-[var(--border)] text-xs text-[var(--muted-foreground)] max-w-md animate-fade-in shadow-md">
          <p className="font-semibold text-[var(--foreground)] mb-1">
            {citation.filename}
            {citation.page_number ? ` — Page ${citation.page_number}` : ""}
          </p>
          <p className="whitespace-pre-wrap leading-relaxed">{citation.content_preview}</p>
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  // 0ms SWR Synchronous Cache Initialization
  const [messages, setMessages] = useState<Message[]>(() => {
    if (typeof window !== "undefined" && sessionId) {
      try {
        const cached = localStorage.getItem(`rag_msgs_${sessionId}`);
        if (cached) {
          const parsed = JSON.parse(cached);
          if (Array.isArray(parsed) && parsed.length > 0) return parsed;
        }
      } catch {}
    }
    return [];
  });

  const [sessionTitle, setSessionTitle] = useState<string>(() => {
    if (typeof window !== "undefined" && sessionId) {
      try {
        const cachedSessions = localStorage.getItem("rag_sessions_cache");
        if (cachedSessions) {
          const list = JSON.parse(cachedSessions);
          const found = list.find((s: any) => s.id === sessionId);
          if (found?.title) return found.title;
        }
      } catch {}
    }
    return "New conversation";
  });

  // Check if session is pinned
  const [isPinned, setIsPinned] = useState<boolean>(() => {
    if (typeof window !== "undefined" && sessionId) {
      try {
        const cached = localStorage.getItem("rag_pinned_sessions");
        if (cached) {
          const list = JSON.parse(cached);
          return Array.isArray(list) && list.includes(sessionId);
        }
      } catch {}
    }
    return false;
  });

  const [input, setInput] = useState<string>(() => {
    if (typeof window !== "undefined" && sessionId) {
      try {
        return localStorage.getItem(`rag_draft_${sessionId}`) || "";
      } catch {}
    }
    return "";
  });

  // Synchronously update input draft when switching between chat sessions
  const [prevSessionId, setPrevSessionId] = useState(sessionId);
  if (prevSessionId !== sessionId) {
    setPrevSessionId(sessionId);
    let draft = "";
    if (typeof window !== "undefined" && sessionId) {
      try {
        draft = localStorage.getItem(`rag_draft_${sessionId}`) || "";
      } catch {}
    }
    setInput(draft);
  }

  const updateDraft = (val: string) => {
    setInput(val);
    if (typeof window !== "undefined" && sessionId) {
      try {
        if (val) {
          localStorage.setItem(`rag_draft_${sessionId}`, val);
        } else {
          localStorage.removeItem(`rag_draft_${sessionId}`);
        }
      } catch {}
    }
  };

  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [error, setError] = useState("");

  // Claude header dropdown menu
  const [isHeaderMenuOpen, setIsHeaderMenuOpen] = useState(false);
  const [copiedShare, setCopiedShare] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Scroll to bottom tracking
  const [showScrollDown, setShowScrollDown] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = (smooth = false) => {
    if (scrollContainerRef.current) {
      if (smooth) {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      } else {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
      }
    }
  };

  const handleScroll = () => {
    const el = scrollContainerRef.current;
    if (el) {
      const isScrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight > 140;
      setShowScrollDown(isScrolledUp);
    }
  };

  useEffect(() => {
    if (!showScrollDown) {
      scrollToBottom(false);
    }
  }, [messages, showScrollDown]);

  // SWR: Instantly load cached messages & revalidate in background via single API call
  useEffect(() => {
    if (typeof window === "undefined" || !sessionId) return;

    // 1. Immediately hydrate from cache on route/session switch
    try {
      const savedDraft = localStorage.getItem(`rag_draft_${sessionId}`) || "";
      setInput(savedDraft);

      const cachedPinned = localStorage.getItem("rag_pinned_sessions");
      if (cachedPinned) {
        const list = JSON.parse(cachedPinned);
        setIsPinned(Array.isArray(list) && list.includes(sessionId));
      }

      const cached = localStorage.getItem(`rag_msgs_${sessionId}`);
      if (cached) {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
        }
      }

      const cachedSessions = localStorage.getItem("rag_sessions_cache");
      if (cachedSessions) {
        const list = JSON.parse(cachedSessions);
        const found = list.find((s: any) => s.id === sessionId);
        if (found?.title) setSessionTitle(found.title);
      }
    } catch {}

    // 2. Fetch fresh messages + title in a single fast backend roundtrip
    let cancelled = false;
    const loadData = async () => {
      try {
        const { messages: freshMsgs, title } = await getChatMessagesWithMetadata(sessionId);
        if (cancelled) return;

        if (title) {
          setSessionTitle(title);
        }
        if (Array.isArray(freshMsgs)) {
          setMessages(freshMsgs);
          try {
            localStorage.setItem(`rag_msgs_${sessionId}`, JSON.stringify(freshMsgs));
          } catch {}
        }
      } catch {
        // Session may be newly created or unsaved
      }
    };

    loadData();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // Close header menu on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-header-dropdown]")) {
        setIsHeaderMenuOpen(false);
      }
    };
    window.addEventListener("click", handleOutsideClick);
    return () => window.removeEventListener("click", handleOutsideClick);
  }, []);

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState("");

  const handleRenameTitle = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = editedTitle.trim();
    if (!trimmed || trimmed === sessionTitle) {
      setIsEditingTitle(false);
      return;
    }

    setSessionTitle(trimmed);
    setIsEditingTitle(false);

    try {
      const cachedSessions = localStorage.getItem("rag_sessions_cache");
      if (cachedSessions) {
        const list = JSON.parse(cachedSessions);
        const updated = list.map((s: any) =>
          s.id === sessionId ? { ...s, title: trimmed } : s
        );
        localStorage.setItem("rag_sessions_cache", JSON.stringify(updated));
      }
    } catch {}

    window.dispatchEvent(new Event("chat-sessions-changed"));

    try {
      await updateChatSession(sessionId, trimmed);
    } catch {
      // Silently ignore or let SWR revalidate
    }
  };

  const handleTogglePin = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setIsHeaderMenuOpen(false);
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_pinned_sessions");
        const list: string[] = cached ? JSON.parse(cached) : [];
        const nextPinned = !isPinned;
        const updated = nextPinned
          ? [sessionId, ...list.filter((id) => id !== sessionId)]
          : list.filter((id) => id !== sessionId);
        localStorage.setItem("rag_pinned_sessions", JSON.stringify(updated));
        setIsPinned(nextPinned);
        window.dispatchEvent(new Event("chat-sessions-changed"));
      } catch {}
    }
  };

  const handleShare = () => {
    if (typeof window !== "undefined") {
      navigator.clipboard.writeText(window.location.href);
      setCopiedShare(true);
      setTimeout(() => setCopiedShare(false), 2000);
    }
  };

  const handleCopyMessage = (msgId: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedMessageId(msgId);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleDeleteChat = async () => {
    setDeleting(true);
    try {
      localStorage.removeItem(`rag_msgs_${sessionId}`);
      localStorage.removeItem(`rag_draft_${sessionId}`);
      const cachedPinned = localStorage.getItem("rag_pinned_sessions");
      if (cachedPinned) {
        const list: string[] = JSON.parse(cachedPinned);
        localStorage.setItem("rag_pinned_sessions", JSON.stringify(list.filter((id) => id !== sessionId)));
      }
    } catch {}

    window.dispatchEvent(
      new CustomEvent("chat-session-deleted", { detail: { sessionId } })
    );
    router.push("/dashboard");

    try {
      await deleteChatSession(sessionId);
    } catch {
      setError("Failed to delete chat session.");
      setDeleting(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || streaming) return;

    const question = input.trim();
    updateDraft("");
    setError("");

    // Add user message
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };
    setMessages((prev) => [...prev, userMsg]);

    // Add empty assistant message for streaming
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
    };
    setMessages((prev) => [...prev, assistantMsg]);
    setStreaming(true);

    try {
      const response = await sendQuery(sessionId, question);
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No response stream");

      let fullContent = "";
      let buffer = "";

      const processSseChunk = (chunkText: string) => {
        const trimmed = chunkText.trim();
        if (!trimmed || trimmed.startsWith(":")) return; // skip empty blocks or comments/pings

        let eventType = "message";
        let dataStr = "";

        for (const rawLine of trimmed.split("\n")) {
          const line = rawLine.trim();
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr) return;

        try {
          const data = JSON.parse(dataStr);

          if (eventType === "token") {
            fullContent += data.token || "";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: fullContent } : m
              )
            );
          } else if (eventType === "citations" || eventType === "citation") {
            const citationsList = Array.isArray(data.citations)
              ? data.citations
              : Array.isArray(data)
              ? data
              : [data];
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, citations: citationsList } : m
              )
            );
          } else if (eventType === "sql_query" || eventType === "sql") {
            const sqlQuery = data.sql || data.query || "";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, sql_query: sqlQuery } : m
              )
            );
          } else if (eventType === "title") {
            const newTitle = data.title;
            if (newTitle) {
              setSessionTitle(newTitle);
              try {
                const cachedSessions = localStorage.getItem("rag_sessions_cache");
                if (cachedSessions) {
                  const list = JSON.parse(cachedSessions);
                  const updated = list.map((s: any) =>
                    s.id === sessionId ? { ...s, title: newTitle } : s
                  );
                  localStorage.setItem("rag_sessions_cache", JSON.stringify(updated));
                }
              } catch {}
              window.dispatchEvent(new Event("chat-sessions-changed"));
            }
          } else if (eventType === "error") {
            setError(
              data.message ||
                "Something went wrong. Please try again."
            );
          }
        } catch {
          // Ignore JSON parse errors for non-JSON or partial data
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Normalize \r\n to \n to ensure CRLF never prevents SSE blocks from splitting
        buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
        
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          processSseChunk(chunk);
        }
      }

      // Flush any trailing buffer
      const finalRemaining = (buffer + decoder.decode()).replace(/\r\n/g, "\n").trim();
      if (finalRemaining) {
        for (const chunk of finalRemaining.split("\n\n")) {
          processSseChunk(chunk);
        }
      }
    } catch (err: any) {
      setError(
        "Something went wrong generating a response. Please try again in a moment."
      );
    } finally {
      setStreaming(false);
      setMessages((latest) => {
        if (typeof window !== "undefined" && sessionId) {
          try {
            localStorage.setItem(`rag_msgs_${sessionId}`, JSON.stringify(latest));
          } catch {}
        }
        return latest;
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[var(--background)]">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--muted-foreground)]" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-[var(--background)] relative">
      {/* Claude-style Chat Header */}
      <div className="h-14 border-b border-[var(--border)]/60 px-6 flex items-center justify-between bg-[var(--background)]/90 backdrop-blur-xs flex-shrink-0 z-10">
        <div className="flex items-center gap-2 min-w-0 flex-1 mr-4">
          {isEditingTitle ? (
            <form onSubmit={handleRenameTitle} className="flex items-center gap-1.5 min-w-0 max-w-sm">
              <input
                type="text"
                value={editedTitle}
                onChange={(e) => setEditedTitle(e.target.value)}
                className="text-sm font-medium px-2 py-0.5 rounded-lg border border-[var(--primary)] bg-[var(--card)] text-[var(--foreground)] focus:outline-none w-full"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Escape") setIsEditingTitle(false);
                }}
              />
              <button
                type="submit"
                title="Save title"
                className="p-1 text-emerald-400 hover:text-emerald-300 transition-colors cursor-pointer"
              >
                <Check className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsEditingTitle(false)}
                title="Cancel"
                className="p-1 text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </form>
          ) : (
            /* Title with Claude-style chevron dropdown */
            <div className="relative" data-header-dropdown>
              <button
                onClick={() => setIsHeaderMenuOpen(!isHeaderMenuOpen)}
                className="flex items-center gap-1.5 px-2 py-1 -ml-2 rounded-lg text-sm font-medium text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer group max-w-md"
              >
                <span className="truncate">{sessionTitle}</span>
                <ChevronDown className={`w-3.5 h-3.5 text-[var(--muted-foreground)] group-hover:text-[var(--foreground)] transition-transform ${isHeaderMenuOpen ? "rotate-180" : ""}`} />
              </button>

              {/* Header Dropdown Menu */}
              {isHeaderMenuOpen && (
                <div className="absolute left-0 top-full mt-1 w-44 bg-[var(--card)] border border-[var(--border)] rounded-xl shadow-xl py-1 z-50 animate-fade-in">
                  <button
                    onClick={handleTogglePin}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors text-left cursor-pointer"
                  >
                    {isPinned ? (
                      <>
                        <PinOff className="w-3.5 h-3.5 text-[var(--muted-foreground)]" />
                        <span>Unpin from top</span>
                      </>
                    ) : (
                      <>
                        <Pin className="w-3.5 h-3.5 text-[var(--primary)]" />
                        <span>Pin to top</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => {
                      setIsHeaderMenuOpen(false);
                      setEditedTitle(sessionTitle);
                      setIsEditingTitle(true);
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors text-left cursor-pointer"
                  >
                    <Pencil className="w-3.5 h-3.5 text-[var(--muted-foreground)]" />
                    <span>Rename chat</span>
                  </button>

                  <div className="my-1 border-t border-[var(--border)]/60" />

                  <button
                    onClick={() => {
                      setIsHeaderMenuOpen(false);
                      setShowDeleteModal(true);
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 transition-colors text-left cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Delete chat</span>
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Header Right Actions: Share */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleShare}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--foreground)] bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 border border-[var(--border)] transition-colors cursor-pointer"
          >
            {copiedShare ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Share2 className="w-3.5 h-3.5 text-[var(--muted-foreground)]" />
                <span>Share</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Messages Thread */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-8 relative"
      >
        <div className="max-w-3xl mx-auto space-y-7">
          {messages.length === 0 && (
            /* Claude-style clean empty state */
            <div className="text-center py-20 animate-fade-in max-w-xl mx-auto">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-[var(--primary)]/10 text-[var(--primary)] mb-5">
                <Sparkles className="w-6 h-6" />
              </div>
              <h2 className="text-xl font-serif font-medium tracking-tight text-[var(--foreground)]">
                Start a document inquiry
              </h2>
              <p className="text-sm text-[var(--muted-foreground)] mt-2 leading-relaxed">
                Ask questions about your uploaded PDFs, reports, spreadsheets, and notes with citations.
              </p>
              <div className="mt-8 flex flex-wrap gap-2 justify-center">
                {[
                  "What are the key findings?",
                  "Summarize the main points",
                  "What is the total revenue?",
                  "Compare quarterly figures",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => updateDraft(q)}
                    className="px-3.5 py-2 rounded-xl text-xs border border-[var(--border)] bg-[var(--card)]/40 text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:border-neutral-500 hover:bg-[var(--secondary)] transition-all cursor-pointer shadow-xs"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col group ${
                msg.role === "user" ? "items-end" : "items-start"
              }`}
            >
              {msg.role === "user" ? (
                /* User Message: Elevated rounded pill on the right */
                <div className="max-w-[82%] rounded-2xl bg-[#2b2b2e] text-[var(--foreground)] px-4 py-2.5 shadow-xs">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">
                    {msg.content}
                  </p>
                </div>
              ) : (
                /* Assistant Message: Claude Document Flow */
                <div className="w-full space-y-3">
                  {msg.sql_query && (
                    <details className="mb-2 group/sql">
                      <summary className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 cursor-pointer hover:bg-amber-500/15 transition-colors select-none">
                        <FileSpreadsheet className="w-3 h-3" />
                        <span>SQL Query</span>
                        <ChevronDown className="w-3 h-3 group-open/sql:hidden" />
                        <ChevronUp className="w-3 h-3 hidden group-open/sql:inline" />
                      </summary>
                      <pre className="mt-2 p-3 rounded-xl bg-[var(--card)] border border-[var(--border)] text-xs text-[var(--muted-foreground)] overflow-x-auto font-mono whitespace-pre-wrap">
                        {msg.sql_query}
                      </pre>
                    </details>
                  )}

                  {/* Thinking / Searching indicator during pre-retrieval */}
                  {streaming && msg.id.startsWith("assistant-") && !msg.content && (
                    <div className="flex items-center gap-2.5 py-1 text-[var(--muted-foreground)]">
                      <div className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] animate-pulse" style={{ animationDelay: "0ms" }} />
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] animate-pulse" style={{ animationDelay: "200ms" }} />
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--primary)] animate-pulse" style={{ animationDelay: "400ms" }} />
                      </div>
                      <span className="text-xs font-medium text-[var(--muted-foreground)]">
                        Searching documents & thinking...
                      </span>
                    </div>
                  )}

                  {msg.content && (
                    <div className="text-sm leading-relaxed text-[var(--foreground)] space-y-2">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={markdownComponents}
                      >
                        {formatMessageContent(msg.content, msg.citations)}
                      </ReactMarkdown>

                      {streaming && msg.id.startsWith("assistant-") && (
                        <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-[var(--primary)] animate-pulse rounded-xs" />
                      )}
                    </div>
                  )}

                  {/* Citations */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="flex flex-wrap gap-2 pt-2">
                      {msg.citations.map((citation, i) => (
                        <CitationChip
                          key={citation.chunk_id}
                          citation={citation}
                          index={i}
                        />
                      ))}
                    </div>
                  )}

                  {/* Action toolbar on assistant reply */}
                  {msg.content && !streaming && (
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 pt-1 text-[var(--muted-foreground)] text-xs">
                      <button
                        onClick={() => handleCopyMessage(msg.id, msg.content)}
                        title="Copy answer"
                        className="p-1 rounded hover:text-[var(--foreground)] transition-colors cursor-pointer flex items-center gap-1"
                      >
                        {copiedMessageId === msg.id ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-[11px] text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span className="text-[11px]">Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {error && (
            <div className="p-3.5 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400 animate-fade-in">
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Scroll to Bottom Button */}
      {showScrollDown && (
        <button
          onClick={() => scrollToBottom(true)}
          title="Scroll to bottom"
          aria-label="Scroll to bottom"
          className="absolute bottom-28 right-8 z-30 p-2.5 rounded-full bg-[var(--card)] border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] shadow-xl transition-all hover:scale-105 cursor-pointer"
        >
          <ArrowDown className="w-4 h-4" />
        </button>
      )}

      {/* Claude Signature Floating Prompt Bar */}
      <div className="px-4 pb-4 pt-2 bg-gradient-to-t from-[var(--background)] via-[var(--background)]/90 to-transparent">
        <div className="max-w-3xl mx-auto">
          {/* Elevated Rounded Input Container */}
          <div className="rounded-2xl bg-[var(--card)] border border-[var(--border)] shadow-xl p-3 focus-within:border-neutral-500/80 transition-all">
            <div className="flex items-end gap-2.5">
              {/* Attach Context (+) Button */}
              <button
                type="button"
                onClick={() => router.push("/dashboard")}
                title="View & Upload Documents"
                className="p-2 rounded-xl text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer flex-shrink-0 mb-0.5"
              >
                <Plus className="w-4 h-4" />
              </button>

              {/* Textarea */}
              <textarea
                value={input}
                onChange={(e) => updateDraft(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Write a message..."
                rows={1}
                className="flex-1 resize-none bg-transparent px-1 py-1.5 text-sm outline-none placeholder:text-[var(--muted-foreground)] text-[var(--foreground)] leading-relaxed max-h-36 overflow-y-auto"
                disabled={streaming}
              />

              {/* Voice / Mic Indicator (Claude style) */}
              <button
                type="button"
                className="p-2 rounded-xl text-[var(--muted-foreground)]/60 hover:text-[var(--muted-foreground)] transition-colors cursor-pointer flex-shrink-0 hidden sm:block mb-0.5"
                title="Voice input (coming soon)"
              >
                <Mic className="w-4 h-4" />
              </button>

              {/* Send Button (Claude Arrow Up pill) */}
              <button
                onClick={handleSend}
                disabled={streaming || !input.trim()}
                title="Send message"
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-all flex-shrink-0 mb-0.5 cursor-pointer ${
                  input.trim() && !streaming
                    ? "bg-[var(--primary)] text-white hover:opacity-90 shadow-sm"
                    : "bg-[var(--secondary)] text-[var(--muted-foreground)]/50 cursor-not-allowed"
                }`}
              >
                {streaming ? (
                  <Loader2 className="w-4 h-4 animate-spin text-[var(--foreground)]" />
                ) : (
                  <ArrowUp className="w-4 h-4 stroke-[2.5]" />
                )}
              </button>
            </div>
          </div>

          {/* Bottom Footer Metadata */}
          <div className="mt-2 px-2 flex items-center justify-between text-[11px] text-[var(--muted-foreground)]/70">
            <p>DocuChat is AI and can make mistakes. Please double-check responses.</p>
            <div className="flex items-center gap-1 font-mono text-[10px] bg-[var(--secondary)]/60 px-2 py-0.5 rounded-full border border-[var(--border)]/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>Hybrid RAG • High</span>
            </div>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4 animate-fade-in">
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl max-w-sm w-full p-6 shadow-2xl animate-fade-in">
            <div className="w-10 h-10 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center mb-4">
              <Trash2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[var(--foreground)]">
              Delete this conversation?
            </h3>
            <p className="text-sm text-[var(--muted-foreground)] mt-2 leading-relaxed">
              Are you sure you want to delete <span className="font-medium text-[var(--foreground)]">&ldquo;{sessionTitle}&rdquo;</span>? All messages in this conversation will be permanently removed.
            </p>
            <div className="flex justify-end gap-2.5 mt-6">
              <button
                onClick={() => setShowDeleteModal(false)}
                disabled={deleting}
                className="px-3.5 py-2 rounded-xl text-sm font-medium border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteChat}
                disabled={deleting}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium bg-red-600 hover:bg-red-500 text-white shadow-sm transition-colors disabled:opacity-50 cursor-pointer"
              >
                {deleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <span>Delete</span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
