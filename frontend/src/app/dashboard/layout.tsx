"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import {
  FileText,
  MessageSquare,
  Plus,
  LogOut,
  Loader2,
  Trash2,
} from "lucide-react";
import {
  getChatSessions,
  getChatMessagesWithMetadata,
  createChatSession,
  deleteChatSession,
  checkBackendHealth,
} from "@/lib/api";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sessions, setSessions] = useState<any[]>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_sessions_cache");
        if (cached) return JSON.parse(cached);
      } catch {}
    }
    return [];
  });
  const [backendAwake, setBackendAwake] = useState<boolean | null>(true);
  const [loadingSessions, setLoadingSessions] = useState(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_sessions_cache");
        if (cached !== null) return false;
      } catch {}
    }
    return true;
  });
  const [sessionToDelete, setSessionToDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  // Check backend health
  useEffect(() => {
    let interval: NodeJS.Timeout;

    const check = async () => {
      const health = await checkBackendHealth();
      setBackendAwake(health.awake && health.modelLoaded);

      if (!health.awake || !health.modelLoaded) {
        interval = setInterval(async () => {
          const h = await checkBackendHealth();
          if (h.awake && h.modelLoaded) {
            setBackendAwake(true);
            clearInterval(interval);
          }
        }, 3000);
      }
    };

    check();
    return () => clearInterval(interval);
  }, []);

  // Load chat sessions immediately (don't wait for health check)
  const loadSessions = useCallback(async () => {
    try {
      const data = await getChatSessions();
      setSessions(data);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sessions_cache", JSON.stringify(data));
        } catch {}
      }
    } catch {
      // Silently fail — sessions will load when backend is ready
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Listen for session changes / deletions from other components
  useEffect(() => {
    const onSessionsChanged = () => {
      loadSessions();
    };
    const onSessionDeleted = (e: any) => {
      const deletedId = e.detail?.sessionId;
      if (deletedId) {
        setSessions((prev) => prev.filter((s) => s.id !== deletedId));
        if (typeof window !== "undefined") {
          try {
            localStorage.removeItem(`rag_draft_${deletedId}`);
            localStorage.removeItem(`rag_msgs_${deletedId}`);
          } catch {}
        }
      }
    };

    window.addEventListener("chat-sessions-changed", onSessionsChanged);
    window.addEventListener("chat-session-deleted", onSessionDeleted as EventListener);

    return () => {
      window.removeEventListener("chat-sessions-changed", onSessionsChanged);
      window.removeEventListener(
        "chat-session-deleted",
        onSessionDeleted as EventListener
      );
    };
  }, [loadSessions]);

  const handleNewChat = async () => {
    // Optimistic: navigate instantly, create session in background
    const tempId = crypto.randomUUID();
    const tempSession = { id: tempId, title: "New conversation", created_at: new Date().toISOString() };
    setSessions((prev) => [tempSession, ...prev]);
    router.push(`/chat/${tempId}`);

    try {
      const realSession = await createChatSession();
      // If user started typing while tempId was active, migrate draft to realSession.id
      if (typeof window !== "undefined") {
        try {
          const tempDraft = localStorage.getItem(`rag_draft_${tempId}`);
          if (tempDraft) {
            localStorage.setItem(`rag_draft_${realSession.id}`, tempDraft);
            localStorage.removeItem(`rag_draft_${tempId}`);
          }
        } catch {}
      }
      // Replace temp session with real one
      setSessions((prev) =>
        prev.map((s) => (s.id === tempId ? realSession : s))
      );
      // Navigate to the real session ID (fast replace, no full reload)
      router.replace(`/chat/${realSession.id}`);
    } catch {
      if (typeof window !== "undefined") {
        try {
          localStorage.removeItem(`rag_draft_${tempId}`);
        } catch {}
      }
      // Remove temp session on failure
      setSessions((prev) => prev.filter((s) => s.id !== tempId));
      router.push("/dashboard");
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    // Optimistic delete — update UI immediately
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== sessionId);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_sessions_cache", JSON.stringify(updated));
          localStorage.removeItem(`rag_draft_${sessionId}`);
          localStorage.removeItem(`rag_msgs_${sessionId}`);
        } catch {}
      }
      return updated;
    });
    setSessionToDelete(null);

    // Navigate away if we're viewing the deleted session
    if (pathname === `/chat/${sessionId}`) {
      router.push("/dashboard");
    }

    // Fire-and-forget: delete in the background
    window.dispatchEvent(
      new CustomEvent("chat-session-deleted", { detail: { sessionId } })
    );
    try {
      await deleteChatSession(sessionId);
    } catch (err) {
      console.error("Failed to delete chat session:", err);
      // Re-fetch sessions to restore state on failure
      loadSessions();
    }
  };

  const handleLogout = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/login");
    router.refresh();
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-[var(--sidebar)] border-r border-[var(--border)] flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-[var(--border)]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[var(--primary)] flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-lg">DocuChat</span>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {/* Documents link */}
          <button
            onClick={() => router.push("/dashboard")}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              pathname === "/dashboard"
                ? "bg-[var(--primary)]/10 text-[var(--primary)]"
                : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]"
            }`}
          >
            <FileText className="w-4 h-4" />
            Documents
          </button>

          {/* New chat button */}
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)] transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>

          {/* Chat sessions */}
          {sessions.length > 0 && (
            <div className="mt-4 pt-4 border-t border-[var(--border)]">
              <p className="px-3 mb-2 text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wider">
                Conversations
              </p>
              <div className="space-y-1">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`group relative flex items-center rounded-lg text-sm transition-colors ${
                      pathname === `/chat/${session.id}`
                        ? "bg-[var(--primary)]/10 text-[var(--primary)]"
                        : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]"
                    }`}
                  >
                    <button
                      onClick={() => router.push(`/chat/${session.id}`)}
                      onMouseEnter={() => {
                        router.prefetch(`/chat/${session.id}`);
                        if (typeof window !== "undefined" && !localStorage.getItem(`rag_msgs_${session.id}`)) {
                          getChatMessagesWithMetadata(session.id)
                            .then(({ messages }) => {
                              if (Array.isArray(messages) && messages.length > 0) {
                                localStorage.setItem(`rag_msgs_${session.id}`, JSON.stringify(messages));
                              }
                            })
                            .catch(() => {});
                        }
                      }}
                      className="flex-1 flex items-center gap-2.5 px-3 py-2 text-left truncate min-w-0 cursor-pointer"
                    >
                      <MessageSquare className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{session.title}</span>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSessionToDelete({ id: session.id, title: session.title });
                      }}
                      title="Delete conversation"
                      aria-label="Delete conversation"
                      className="opacity-0 group-hover:opacity-100 p-1.5 mr-1.5 rounded-md text-[var(--muted-foreground)] hover:text-red-400 hover:bg-red-500/10 transition-all flex-shrink-0 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </nav>

        {/* Logout */}
        <div className="p-3 border-t border-[var(--border)]">
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)] transition-colors cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Backend wake-up banner */}
        {backendAwake === false && (
          <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-3 flex items-center gap-3 animate-fade-in">
            <div className="flex gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "300ms" }} />
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse-dot" style={{ animationDelay: "600ms" }} />
            </div>
            <p className="text-sm text-amber-300">
              The backend is waking up — this can take up to a minute on the free tier. Hang tight!
            </p>
          </div>
        )}

        {children}
      </main>

      {/* Delete Conversation Modal */}
      {sessionToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 animate-fade-in">
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-2xl max-w-sm w-full p-6 shadow-2xl animate-fade-in">
            <div className="w-10 h-10 rounded-xl bg-red-500/10 text-red-400 flex items-center justify-center mb-4">
              <Trash2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[var(--foreground)]">
              Delete conversation?
            </h3>
            <p className="text-sm text-[var(--muted-foreground)] mt-2 leading-relaxed">
              Are you sure you want to delete <span className="font-medium text-[var(--foreground)]">&ldquo;{sessionToDelete.title}&rdquo;</span>? All messages in this conversation will be permanently removed.
            </p>
            <div className="flex justify-end gap-2.5 mt-6">
              <button
                onClick={() => setSessionToDelete(null)}
                className="px-3.5 py-2 rounded-xl text-sm font-medium border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--secondary)] transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteSession(sessionToDelete.id)}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium bg-red-600 hover:bg-red-500 text-white shadow-sm transition-colors cursor-pointer"
              >
                <span>Delete</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
