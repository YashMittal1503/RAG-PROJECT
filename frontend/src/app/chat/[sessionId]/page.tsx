"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  getChatMessages,
  getChatSession,
  deleteChatSession,
  sendQuery,
} from "@/lib/api";
import {
  Send,
  Loader2,
  FileText,
  FileSpreadsheet,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Sparkles,
  Trash2,
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
    if (inline) {
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
      <div className="relative my-3 rounded-xl overflow-hidden border border-[var(--border)] bg-[#0d1117]">
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
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-[var(--primary)]/15 text-[var(--primary)] hover:bg-[var(--primary)]/25 transition-colors border border-[var(--primary)]/30 shadow-sm"
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

  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionTitle, setSessionTitle] = useState<string>("Conversation");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load existing messages (critical path — load first)
  useEffect(() => {
    const load = async () => {
      try {
        const msgs = await getChatMessages(sessionId);
        setMessages(msgs);
      } catch {
        // Session might not exist yet
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [sessionId]);

  // Load session title in background (non-critical)
  useEffect(() => {
    getChatSession(sessionId)
      .then((data) => {
        if (data?.title) setSessionTitle(data.title);
      })
      .catch(() => {});
  }, [sessionId]);

  const handleDeleteChat = async () => {
    setDeleting(true);
    try {
      await deleteChatSession(sessionId);
      window.dispatchEvent(
        new CustomEvent("chat-session-deleted", { detail: { sessionId } })
      );
      router.push("/dashboard");
    } catch {
      setError("Failed to delete chat session.");
      setDeleting(false);
      setShowDeleteModal(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || streaming) return;

    const question = input.trim();
    setInput("");
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

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // SSE messages are separated by double newline \n\n
        const messages = buffer.split("\n\n");
        // Keep the last potentially incomplete chunk in the buffer
        buffer = messages.pop() || "";

        for (const message of messages) {
          if (!message.trim()) continue;

          let eventType = "message";
          let dataStr = "";

          for (const line of message.split("\n")) {
            const trimmed = line.trim();
            if (trimmed.startsWith("event:")) {
              eventType = trimmed.slice(6).trim();
            } else if (trimmed.startsWith("data:")) {
              dataStr = trimmed.slice(5).trim();
            }
          }

          if (!dataStr || dataStr === "{}") continue;

          try {
            const data = JSON.parse(dataStr);

            if (eventType === "token" && data.token) {
              fullContent += data.token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: fullContent }
                    : m
                )
              );
            } else if (eventType === "citations" && data.citations) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, citations: data.citations }
                    : m
                )
              );
            } else if (eventType === "error") {
              setError(
                data.message ||
                  "Something went wrong. Please try again."
              );
            }
          } catch {
            // Ignore JSON parse errors for incomplete data
          }
        }
      }
    } catch (err: any) {
      setError(
        "Something went wrong generating a response. Please try again in a moment."
      );
    } finally {
      setStreaming(false);
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
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--muted-foreground)]" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Chat header */}
      <div className="border-b border-[var(--border)] px-6 py-3 flex items-center justify-between bg-[var(--background)]/80 backdrop-blur-xs flex-shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-[var(--primary)]/10 text-[var(--primary)] flex items-center justify-center flex-shrink-0">
            <MessageSquare className="w-3.5 h-3.5" />
          </div>
          <h2 className="text-sm font-semibold truncate text-[var(--foreground)]">
            {sessionTitle}
          </h2>
        </div>
        <button
          onClick={() => setShowDeleteModal(true)}
          title="Delete this conversation"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[var(--muted-foreground)] hover:text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-colors cursor-pointer"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Delete chat</span>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            /* Empty state */
            <div className="text-center py-20 animate-fade-in">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-[var(--primary)]/10 mb-4">
                <Sparkles className="w-8 h-8 text-[var(--primary)]" />
              </div>
              <h3 className="text-lg font-medium">
                Ask a question about your documents
              </h3>
              <p className="text-[var(--muted-foreground)] mt-2 max-w-md mx-auto">
                I&apos;ll search through your uploaded documents and provide
                answers with source citations.
              </p>
              <div className="mt-6 flex flex-wrap gap-2 justify-center">
                {[
                  "What are the key findings?",
                  "Summarize the main points",
                  "What's the total revenue?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setInput(q);
                    }}
                    className="px-3 py-1.5 rounded-lg text-sm border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:border-[var(--muted-foreground)] transition-colors"
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
              className={`flex gap-3 animate-slide-up ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-lg bg-[var(--primary)] flex items-center justify-center flex-shrink-0 mt-1">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
              )}

              <div
                className={`max-w-[80%] ${
                  msg.role === "user"
                    ? "rounded-2xl rounded-br-md bg-[var(--primary)] text-white px-4 py-3"
                    : "space-y-3"
                }`}
              >
                {/* Message content */}
                <div
                  className={`${
                    msg.role === "assistant"
                      ? "rounded-2xl rounded-bl-md bg-[var(--card)] border border-[var(--border)] px-4 py-3 shadow-xs"
                      : ""
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                    </p>
                  ) : (
                    <div className="relative">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={markdownComponents}
                      >
                        {formatMessageContent(msg.content, msg.citations)}
                      </ReactMarkdown>
                      {streaming &&
                        msg.id.startsWith("assistant-") &&
                        msg.content.length > 0 && (
                          <span className="inline-block w-2 h-4 ml-1 align-middle bg-[var(--primary)] animate-pulse rounded-xs" />
                        )}
                    </div>
                  )}
                </div>

                {/* Citations */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="flex flex-wrap gap-2 px-1">
                    {msg.citations.map((citation, i) => (
                      <CitationChip
                        key={citation.chunk_id}
                        citation={citation}
                        index={i}
                      />
                    ))}
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-lg bg-[var(--secondary)] flex items-center justify-center flex-shrink-0 mt-1">
                  <MessageSquare className="w-4 h-4 text-[var(--muted-foreground)]" />
                </div>
              )}
            </div>
          ))}

          {/* Error message */}
          {error && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400 animate-fade-in">
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="border-t border-[var(--border)] p-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 text-sm outline-none transition-colors focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary)] placeholder:text-[var(--muted-foreground)]"
            disabled={streaming}
          />
          <button
            onClick={handleSend}
            disabled={streaming || !input.trim()}
            className="rounded-xl bg-[var(--primary)] px-4 py-3 text-white transition-all hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {streaming ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 animate-fade-in">
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
