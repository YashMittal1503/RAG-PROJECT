"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  getDocuments,
  deleteDocument,
  uploadFiles,
  getDocumentStatus,
} from "@/lib/api";
import {
  Upload,
  Trash2,
  FileText,
  FileSpreadsheet,
  File,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  X,
} from "lucide-react";

type Document = {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  failure_reason?: string;
  chunk_count?: number;
  created_at: string;
};

const STATUS_CONFIG: Record<
  string,
  { icon: React.ReactNode; label: string; color: string }
> = {
  queued: {
    icon: <Clock className="w-3.5 h-3.5" />,
    label: "Queued",
    color: "text-gray-400 bg-gray-400/10",
  },
  parsing: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "Parsing",
    color: "text-blue-400 bg-blue-400/10",
  },
  chunking: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "Chunking",
    color: "text-blue-400 bg-blue-400/10",
  },
  embedding: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "Embedding",
    color: "text-indigo-400 bg-indigo-400/10",
  },
  storing: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "Storing",
    color: "text-violet-400 bg-violet-400/10",
  },
  ready: {
    icon: <CheckCircle2 className="w-3.5 h-3.5" />,
    label: "Ready",
    color: "text-green-400 bg-green-400/10",
  },
  failed: {
    icon: <XCircle className="w-3.5 h-3.5" />,
    label: "Failed",
    color: "text-red-400 bg-red-400/10",
  },
};

function FileIcon({ type }: { type: string }) {
  switch (type) {
    case "pdf":
      return <FileText className="w-5 h-5 text-red-400" />;
    case "xlsx":
    case "csv":
      return <FileSpreadsheet className="w-5 h-5 text-green-400" />;
    default:
      return <File className="w-5 h-5 text-blue-400" />;
  }
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DashboardPage() {
  // Initialize from localStorage if available for instant 0ms render
  const [documents, setDocuments] = useState<Document[]>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_documents_cache");
        if (cached) return JSON.parse(cached);
      } catch {}
    }
    return [];
  });
  // If we already have cached documents, render immediately without blocking spinner
  const [loading, setLoading] = useState<boolean>(() => {
    if (typeof window !== "undefined") {
      try {
        const cached = localStorage.getItem("rag_documents_cache");
        if (cached !== null) return false;
      } catch {}
    }
    return true;
  });
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadErrors, setUploadErrors] = useState<{ filename: string; error: string }[]>([]);
  const router = useRouter();

  const loadDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_documents_cache", JSON.stringify(docs));
        } catch {}
      }
    } catch {
      // Will retry on next load
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  // Poll for status updates on processing documents
  useEffect(() => {
    const processingDocs = documents.filter(
      (d) => !["ready", "failed"].includes(d.status)
    );

    if (processingDocs.length === 0) return;

    const interval = setInterval(async () => {
      let updated = false;
      const newDocs = [...documents];

      for (const doc of processingDocs) {
        try {
          const status = await getDocumentStatus(doc.id);
          const idx = newDocs.findIndex((d) => d.id === doc.id);
          if (idx >= 0 && newDocs[idx].status !== status.status) {
            newDocs[idx] = { ...newDocs[idx], ...status };
            updated = true;
          }
        } catch {
          // Ignore polling errors
        }
      }

      if (updated) {
        setDocuments(newDocs);
        if (typeof window !== "undefined") {
          try {
            sessionStorage.setItem("rag_documents_cache", JSON.stringify(newDocs));
          } catch {}
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [documents]);

  const handleUpload = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    setUploading(true);
    setUploadErrors([]);

    try {
      const result = await uploadFiles(files);
      if (result.errors?.length > 0) {
        setUploadErrors(result.errors);
      }
      // Reload documents to show new ones
      await loadDocuments();
    } catch (err: any) {
      setUploadErrors([{ filename: "Upload", error: err.message }]);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      const updated = documents.filter((d) => d.id !== docId);
      setDocuments(updated);
      if (typeof window !== "undefined") {
        try {
          localStorage.setItem("rag_documents_cache", JSON.stringify(updated));
        } catch {}
      }
    } catch {
      // Show error toast in production
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold">Documents</h1>
          <p className="text-[var(--muted-foreground)] mt-1">
            Upload and manage your documents for AI-powered Q&A
          </p>
        </div>

        {/* Upload Zone */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
            dragOver
              ? "border-[var(--primary)] bg-[var(--primary)]/5"
              : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
          }`}
          onClick={() => {
            const input = document.createElement("input");
            input.type = "file";
            input.multiple = true;
            input.accept = ".pdf,.txt,.csv,.xlsx";
            input.onchange = (e) => {
              const files = (e.target as HTMLInputElement).files;
              if (files) handleUpload(files);
            };
            input.click();
          }}
        >
          {uploading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-10 h-10 text-[var(--primary)] animate-spin" />
              <p className="text-sm text-[var(--muted-foreground)]">
                Uploading files...
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Upload className="w-10 h-10 text-[var(--muted-foreground)]" />
              <div>
                <p className="font-medium">
                  Drop files here or click to browse
                </p>
                <p className="text-sm text-[var(--muted-foreground)] mt-1">
                  PDF, TXT, CSV, XLSX — up to 20MB each, 10 files per upload
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Upload Errors */}
        {uploadErrors.length > 0 && (
          <div className="space-y-2 animate-fade-in">
            {uploadErrors.map((err, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3"
              >
                <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1 text-sm">
                  <span className="font-medium text-red-400">
                    {err.filename}:
                  </span>{" "}
                  <span className="text-red-300">{err.error}</span>
                </div>
                <button
                  onClick={() =>
                    setUploadErrors((prev) => prev.filter((_, j) => j !== i))
                  }
                  className="text-red-400 hover:text-red-300"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Document List */}
        {loading ? (
          <div className="space-y-3 py-4">
            <div className="flex items-center justify-center gap-2 text-sm text-[var(--muted-foreground)] py-6">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--primary)]" />
              <span>Loading documents...</span>
            </div>
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="flex items-center gap-4 rounded-xl border border-[var(--border)] bg-[var(--card)]/50 px-4 py-3 animate-pulse"
              >
                <div className="w-10 h-10 rounded-lg bg-[var(--secondary)]" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-[var(--secondary)] rounded w-1/3" />
                  <div className="h-3 bg-[var(--secondary)] rounded w-1/5" />
                </div>
                <div className="h-6 w-16 bg-[var(--secondary)] rounded-full" />
              </div>
            ))}
          </div>
        ) : documents.length === 0 ? (
          /* Empty State */
          <div className="text-center py-16 animate-fade-in">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-[var(--secondary)] mb-4">
              <FileText className="w-8 h-8 text-[var(--muted-foreground)]" />
            </div>
            <h3 className="text-lg font-medium">No documents yet</h3>
            <p className="text-[var(--muted-foreground)] mt-1 max-w-sm mx-auto">
              Upload your first document to get started. You can then ask
              questions about its content in the chat.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => {
              const statusConfig = STATUS_CONFIG[doc.status] || STATUS_CONFIG.queued;
              return (
                <div
                  key={doc.id}
                  className="flex items-center gap-4 rounded-xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 animate-fade-in"
                >
                  <FileIcon type={doc.file_type} />

                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{doc.filename}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">
                      {formatFileSize(doc.file_size)}
                      {doc.chunk_count
                        ? ` · ${doc.chunk_count} chunks`
                        : ""}
                    </p>
                  </div>

                  {/* Status Badge */}
                  <div
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${statusConfig.color}`}
                    title={doc.failure_reason || ""}
                  >
                    {statusConfig.icon}
                    {statusConfig.label}
                  </div>

                  {/* Failure reason tooltip */}
                  {doc.status === "failed" && doc.failure_reason && (
                    <span
                      className="text-xs text-red-400 max-w-[200px] truncate"
                      title={doc.failure_reason}
                    >
                      {doc.failure_reason}
                    </span>
                  )}

                  {/* Delete button */}
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="p-1.5 rounded-lg text-[var(--muted-foreground)] hover:text-red-400 hover:bg-red-400/10 transition-colors"
                    title="Delete document"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
